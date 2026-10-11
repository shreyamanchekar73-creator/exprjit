use std::fs;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use arrow::array::{Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;

/// Load a CSV file at `path` into a `RecordBatch`. The first line is
/// treated as the header (column names); every subsequent line must have
/// the same number of comma-separated fields.
pub fn load_csv(path: impl AsRef<Path>) -> Result<RecordBatch> {
    let path = path.as_ref();
    let content =
        fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    // Files saved by Excel/Notepad may start with a UTF-8 BOM, which would
    // otherwise become part of the first column name.
    let content = content.trim_start_matches('\u{feff}');

    let mut lines = content.lines().filter(|l| !l.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("{}: file is empty (no header row)", path.display()))?;
    let col_names: Vec<String> = header.split(',').map(|s| s.trim().to_string()).collect();
    let mut seen = std::collections::HashSet::new();
    for n in &col_names {
        if n.is_empty() {
            bail!("{}: header has an empty column name", path.display());
        }
        if !seen.insert(n.as_str()) {
            bail!(
                "{}: duplicate column name `{}` in header",
                path.display(),
                n
            );
        }
    }

    // Collect every row as raw string cells first, so we can decide each
    // column's type by looking at *all* its values before committing.
    let mut raw_cols: Vec<Vec<String>> = vec![Vec::new(); col_names.len()];
    for (row_idx, line) in lines.enumerate() {
        let cells: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if cells.len() != col_names.len() {
            bail!(
                "{}: row {} has {} field(s), expected {} (matching the header)",
                path.display(),
                row_idx + 2, // +1 for header, +1 for 1-based row numbering
                cells.len(),
                col_names.len()
            );
        }
        for (col_idx, cell) in cells.iter().enumerate() {
            raw_cols[col_idx].push(cell.to_string());
        }
    }

    let num_rows = raw_cols.first().map(|c| c.len()).unwrap_or(0);
    if num_rows == 0 {
        bail!("{}: no data rows (only a header)", path.display());
    }

    let mut fields = Vec::with_capacity(col_names.len());
    let mut arrays: Vec<Arc<dyn arrow::array::Array>> = Vec::with_capacity(col_names.len());

    for (name, values) in col_names.iter().zip(raw_cols.iter()) {
        let all_ints = values.iter().all(|v| v.parse::<i64>().is_ok());
        if all_ints {
            let parsed: Vec<i64> = values.iter().map(|v| v.parse::<i64>().unwrap()).collect();
            fields.push(Field::new(name, DataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(parsed)));
            continue;
        }
        // NaN / inf parse as f64 but break JIT-vs-interpreter agreement, so reject them.
        let all_floats = values
            .iter()
            .all(|v| v.parse::<f64>().is_ok_and(|x| x.is_finite()));
        if all_floats {
            let parsed: Vec<f64> = values.iter().map(|v| v.parse::<f64>().unwrap()).collect();
            fields.push(Field::new(name, DataType::Float64, false));
            arrays.push(Arc::new(Float64Array::from(parsed)));
            continue;
        }
        // Find the first offending value for a precise error message.
        let bad = values
            .iter()
            .find(|v| !v.parse::<f64>().is_ok_and(|x| x.is_finite()))
            .cloned()
            .unwrap_or_default();
        bail!(
            "{}: column `{name}` has a non-numeric value ({bad:?}) \u{2014} \
             only Int64/Float64 columns are supported by the JIT filter path right now",
            path.display()
        );
    }

    let schema = Arc::new(ArrowSchema::new(fields));
    RecordBatch::try_new(schema, arrays)
        .with_context(|| format!("{}: building RecordBatch", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_csv(contents: &str) -> tempfile_lite::NamedTempFile {
        let mut f = tempfile_lite::NamedTempFile::new();
        f.write_all(contents.as_bytes()).unwrap();
        f
    }

    /// Tiny inline stand-in for `tempfile` (kept dependency-free): writes to
    /// a uniquely-named file under the OS temp dir and deletes it on drop.
    mod tempfile_lite {
        use std::fs::File;
        use std::io::Write;
        use std::path::PathBuf;
        use std::sync::atomic::{AtomicU64, Ordering};

        static COUNTER: AtomicU64 = AtomicU64::new(0);

        pub struct NamedTempFile {
            path: PathBuf,
            file: File,
        }
        impl NamedTempFile {
            pub fn new() -> Self {
                let unique = COUNTER.fetch_add(1, Ordering::SeqCst);
                let path = std::env::temp_dir().join(format!(
                    "exprjit_test_{}_{}_{}.csv",
                    std::process::id(),
                    unique,
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                ));
                let file = File::create(&path).unwrap();
                NamedTempFile { path, file }
            }
            pub fn path(&self) -> &PathBuf {
                &self.path
            }
        }
        impl Write for NamedTempFile {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.file.write(buf)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                self.file.flush()
            }
        }
        impl Drop for NamedTempFile {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.path);
            }
        }
    }

    #[test]
    fn loads_int_and_float_columns() {
        let f = write_temp_csv("age,score\n10,5.5\n20,7.25\n30,1.0\n");
        let batch = load_csv(f.path()).unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.schema().field(0).data_type(), &DataType::Int64);
        assert_eq!(batch.schema().field(1).data_type(), &DataType::Float64);
    }

    #[test]
    fn mixed_int_float_values_promote_column_to_float() {
        // "score" has both integer-looking and fractional values across
        // rows -> the whole column should load as Float64, not error.
        let f = write_temp_csv("score\n5\n5.5\n7\n");
        let batch = load_csv(f.path()).unwrap();
        assert_eq!(batch.schema().field(0).data_type(), &DataType::Float64);
    }

    #[test]
    fn rejects_non_numeric_column() {
        let f = write_temp_csv("name,age\nbob,10\nalice,20\n");
        let err = load_csv(f.path()).unwrap_err();
        assert!(
            err.to_string().contains("name"),
            "error should name the offending column: {err}"
        );
    }

    #[test]
    fn rejects_ragged_rows() {
        let f = write_temp_csv("a,b\n1,2\n3\n");
        assert!(load_csv(f.path()).is_err());
    }

    #[test]
    fn strips_utf8_bom_from_first_header() {
        let f = write_temp_csv("\u{feff}age,score\n1,2.5\n");
        let batch = load_csv(f.path()).unwrap();
        assert_eq!(batch.schema().field(0).name(), "age");
    }

    #[test]
    fn rejects_duplicate_headers() {
        let f = write_temp_csv("a,a\n1,2\n");
        assert!(load_csv(f.path()).is_err());
    }

    #[test]
    fn rejects_nan_and_inf() {
        let f = write_temp_csv("x\n1.5\nNaN\n");
        assert!(load_csv(f.path()).is_err());
    }

    #[test]
    fn rejects_empty_file() {
        let f = write_temp_csv("");
        assert!(load_csv(f.path()).is_err());
    }
}
