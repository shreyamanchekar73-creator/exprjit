use std::collections::HashMap;
use std::env;
use std::fs;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use arrow::array::{Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;

use exprjit::ast::Expr;
use exprjit::csv_loader::load_csv;
use exprjit::engine::Engine;
use exprjit::interp::{self, Value};
use exprjit::parser::parse;

const USAGE: &str = r#"exprjit - JIT-compiled query filtering over Arrow record batches

USAGE:
    exprjit                                Run a demo query
    exprjit "<query>"                      Run a custom query
    exprjit --rows <n> "<query>"           Use n synthetic rows (default 20)
    exprjit --csv <path> "<query>"         Load CSV and run query
    exprjit --csv <path> --list-columns    Show CSV columns
    exprjit --help                         Show this message

QUERY SYNTAX:
    Comparisons: >  <  >=  <=  ==  !=
    Logic:       &&  ||  !
    Values:      column names, integer/float literals
    Grouping:    ( ... )

EXAMPLES:
    age > 18
    age > 10 && score < 20.0
    age > 15 || score > 15.0
    (age > 10) && (score > 5.0)

TIP:
    Use `cargo run --release` for meaningful timings, and --rows 1000000
    so there is enough data for the speedup to show.
"#;

/// Built-in demo data: `rows` rows containing age (0..rows) and score (age * 1.3).
fn sample_batch(rows: usize) -> Result<RecordBatch> {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("age", DataType::Int64, false),
        Field::new("score", DataType::Float64, false),
    ]));

    let age = Int64Array::from((0..rows as i64).collect::<Vec<_>>());

    let score = Float64Array::from((0..rows).map(|i| i as f64 * 1.3).collect::<Vec<_>>());

    RecordBatch::try_new(schema, vec![Arc::new(age), Arc::new(score)])
        .context("building sample batch")
}

/// Display columns and data types.
fn list_columns(batch: &RecordBatch) {
    println!("columns:");

    let schema = batch.schema();

    for field in schema.fields().iter() {
        println!("  {:<20} {:?}", field.name(), field.data_type());
    }

    println!("rows: {}", batch.num_rows());
}

/// Command-line arguments.
struct Args {
    csv_path: Option<String>,
    rows: Option<usize>,
    query: Option<String>,
    list_columns: bool,
    help: bool,
}

/// Parse command-line arguments.
fn parse_args() -> Result<Args> {
    let mut args = Args {
        csv_path: None,
        rows: None,
        query: None,
        list_columns: false,
        help: false,
    };

    let mut it = env::args().skip(1);

    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--csv" => match it.next() {
                Some(p) if !p.starts_with("--") => args.csv_path = Some(p),
                _ => anyhow::bail!("--csv needs a file path, e.g. --csv data.csv"),
            },

            "--rows" => {
                let value = it
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--rows needs a number, e.g. --rows 1000000"))?;
                let n: usize = value.parse().with_context(|| {
                    format!("--rows must be a positive whole number, got `{value}`")
                })?;
                if n == 0 {
                    anyhow::bail!("--rows must be at least 1");
                }
                args.rows = Some(n);
            }

            "--list-columns" => {
                args.list_columns = true;
            }

            "--help" | "-h" => {
                args.help = true;
            }

            other => {
                args.query = Some(other.to_string());
            }
        }
    }

    Ok(args)
}

/// Run the interpreter version of the query.
///
/// Columns are resolved once and a single row map is reused for every row,
/// so the timing reflects expression evaluation rather than per-row
/// allocation.
fn interpreter_filter(batch: &RecordBatch, expr: &Expr) -> Result<(Vec<u32>, std::time::Duration)> {
    enum Col<'a> {
        Int(&'a Int64Array),
        Float(&'a Float64Array),
    }

    let schema = batch.schema();
    let mut cols: Vec<(&str, Col<'_>)> = Vec::with_capacity(schema.fields().len());
    for (idx, field) in schema.fields().iter().enumerate() {
        let array = batch.column(idx);
        let col = match array.data_type() {
            DataType::Int64 => Col::Int(
                array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| anyhow::anyhow!("failed to read Int64 column"))?,
            ),
            DataType::Float64 => Col::Float(
                array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| anyhow::anyhow!("failed to read Float64 column"))?,
            ),
            other => anyhow::bail!("interpreter does not support {other:?}"),
        };
        cols.push((field.name().as_str(), col));
    }

    let started = Instant::now();
    let mut kept = Vec::new();
    let mut row: HashMap<&str, Value> = HashMap::with_capacity(cols.len());

    for row_idx in 0..batch.num_rows() {
        for (name, col) in &cols {
            let value = match col {
                Col::Int(a) => Value::Int(a.value(row_idx)),
                Col::Float(a) => Value::Float(a.value(row_idx)),
            };
            row.insert(*name, value);
        }

        if interp::eval_bool(expr, &row).with_context(|| format!("interpreting row {row_idx}"))? {
            kept.push(row_idx as u32);
        }
    }

    Ok((kept, started.elapsed()))
}

/// Dashboard data.
struct QueryStats {
    query: String,
    total_rows: usize,
    matched_rows: usize,
    match_pct: f64,
    compile_us: f64,
    jit_us: f64,
    interpreter_us: f64,
    speedup: f64,
    cache_status: String,
    correctness: String,
    indices: Vec<u32>,
}

/// Escape text so it is safe inside JavaScript strings.
fn js_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

/// Write the results of this session to dashboard_data.js.
///
/// The top-level fields describe the latest run (as before); `history`
/// lists every run of the session, oldest first. `coldCompileUs` comes
/// from the first run, because a cache hit compiles nothing.
fn write_dashboard_data(runs: &[QueryStats], compiled_shapes: usize) -> Result<()> {
    let (Some(first), Some(latest)) = (runs.first(), runs.last()) else {
        anyhow::bail!("no runs to write to the dashboard");
    };
    let cold_compile_us = first.compile_us;

    // Only a sample of indexes goes to the dashboard: with millions of rows
    // the full list would make the file huge. `matchedRows` has the count.
    let indices = latest
        .indices
        .iter()
        .take(100)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    let history = runs
        .iter()
        .map(|r| {
            format!(
                r#"{{ query: "{}", rows: {}, matched: {}, cache: "{}", jitUs: {:.2}, interpreterUs: {:.2} }}"#,
                js_escape(&r.query),
                r.total_rows,
                r.matched_rows,
                r.cache_status,
                r.jit_us,
                r.interpreter_us
            )
        })
        .collect::<Vec<_>>()
        .join(",\n        ");

    let js = format!(
        r#"window.EXPRJIT_DATA = {{
    query: "{}",
    totalRows: {},
    matchedRows: {},
    matchPct: {:.2},
    coldCompileUs: {:.2},
    warmCompileUs: 0,
    compileUs: {:.2},
    jitUs: {:.2},
    interpreterUs: {:.2},
    speedup: {:.2},
    cacheStatus: "{}",
    correctness: "{}",
    compiledShapes: {},
    indices: [{}],
    history: [
        {}
    ]
}};
"#,
        js_escape(&latest.query),
        latest.total_rows,
        latest.matched_rows,
        latest.match_pct,
        cold_compile_us,
        latest.compile_us,
        latest.jit_us,
        latest.interpreter_us,
        latest.speedup,
        latest.cache_status,
        latest.correctness,
        compiled_shapes,
        indices,
        history
    );

    fs::write("dashboard_data.js", js).context("writing dashboard_data.js")?;

    Ok(())
}

/// Run one query.
fn run_query(
    engine: &mut Engine,
    batch: &RecordBatch,
    query: &str,
    label: &str,
) -> Result<QueryStats> {
    println!();
    println!("Running: {}", label);

    // ------------------------------------------------------------
    // 1. JIT COMPILATION
    // ------------------------------------------------------------

    let compiled = engine
        .compile_query(query, batch)
        .with_context(|| format!("compiling query `{query}`"))?;

    // ------------------------------------------------------------
    // 2. JIT EXECUTION
    // ------------------------------------------------------------

    let started = Instant::now();

    let kept = compiled
        .filter(batch)
        .with_context(|| format!("running query `{query}`"))?;

    let jit_time = started.elapsed();

    // ------------------------------------------------------------
    // 3. INTERPRETER EXECUTION
    // ------------------------------------------------------------

    let expr = parse(query)
        .map_err(|e| anyhow::anyhow!(e.to_string()))
        .context("parsing query for interpreter comparison")?;

    let (interp_kept, interp_time) = interpreter_filter(batch, &expr)?;

    // ------------------------------------------------------------
    // 4. SPEEDUP
    // ------------------------------------------------------------

    let speedup = if jit_time.as_nanos() > 0 {
        interp_time.as_secs_f64() / jit_time.as_secs_f64()
    } else {
        0.0
    };

    // ------------------------------------------------------------
    // 5. CACHE
    // ------------------------------------------------------------

    let cache_status = if compiled.cache_hit { "HIT" } else { "MISS" };

    let compile_status = if compiled.cache_hit {
        "cache hit - reused compiled code".to_string()
    } else {
        format!("compiled in {:.2?}", compiled.compile_time)
    };

    // ------------------------------------------------------------
    // 6. MATCH PERCENTAGE
    // ------------------------------------------------------------

    let match_pct = if batch.num_rows() == 0 {
        0.0
    } else {
        kept.len() as f64 * 100.0 / batch.num_rows() as f64
    };

    // ------------------------------------------------------------
    // 7. CORRECTNESS
    // ------------------------------------------------------------

    let correctness = if kept == interp_kept { "PASS" } else { "FAIL" };

    // Compilation time in microseconds.
    let compile_us = compiled.compile_time.as_secs_f64() * 1_000_000.0;

    let jit_us = jit_time.as_secs_f64() * 1_000_000.0;

    let interpreter_us = interp_time.as_secs_f64() * 1_000_000.0;

    // ------------------------------------------------------------
    // DISPLAY
    // ------------------------------------------------------------

    println!();
    println!("============================================================");
    println!("                    ExprJIT RESULT");
    println!("============================================================");

    println!("Query              : {}", query);

    println!("------------------------------------------------------------");

    println!("Total Rows         : {}", batch.num_rows());

    println!("Rows Matched       : {}", kept.len());

    println!("Match Percentage   : {:.1}%", match_pct);

    println!("------------------------------------------------------------");

    println!("Compilation        : {}", compile_status);

    println!("Cache Status       : {}", cache_status);

    println!("------------------------------------------------------------");

    println!("JIT Execution      : {:.2?}", jit_time);

    println!("Interpreter Time   : {:.2?}", interp_time);

    println!("JIT Speedup        : {:.2}x", speedup);

    println!("------------------------------------------------------------");

    println!("Correctness        : {}", correctness);

    let shown: Vec<u32> = kept.iter().copied().take(20).collect();
    if kept.len() > shown.len() {
        println!(
            "Matched Row Indexes: {:?} ... and {} more",
            shown,
            kept.len() - shown.len()
        );
    } else {
        println!("Matched Row Indexes: {:?}", shown);
    }

    println!("============================================================");

    Ok(QueryStats {
        query: query.to_string(),
        total_rows: batch.num_rows(),
        matched_rows: kept.len(),
        match_pct,
        compile_us,
        jit_us,
        interpreter_us,
        speedup,
        cache_status: cache_status.to_string(),
        correctness: correctness.to_string(),
        indices: kept,
    })
}

/// Main program.
fn run() -> Result<()> {
    let args = parse_args()?;

    // HELP
    if args.help {
        print!("{USAGE}");
        return Ok(());
    }

    if cfg!(debug_assertions) {
        eprintln!(
            "note: debug build - timings are not representative. \
             Use `cargo run --release` for real numbers."
        );
    }

    // LOAD DATA
    let batch = match (&args.csv_path, args.rows) {
        (Some(_), Some(_)) => anyhow::bail!("use either --csv or --rows, not both"),
        (Some(path), None) => load_csv(path).with_context(|| format!("loading CSV from {path}"))?,
        (None, rows) => sample_batch(rows.unwrap_or(20))?,
    };

    // LIST COLUMNS
    if args.list_columns {
        list_columns(&batch);
        return Ok(());
    }

    // QUERY
    let query = args
        .query
        .unwrap_or_else(|| "age > 10 && score < 20.0".to_string());

    // ENGINE
    let mut engine = Engine::new();

    // Run the same query twice: the first run compiles (cache MISS), the
    // second reuses the compiled code (cache HIT).
    let cold = run_query(&mut engine, &batch, &query, "run 1 (cold)")?;
    let warm = run_query(&mut engine, &batch, &query, "run 2 (warm)")?;

    // WRITE DASHBOARD DATA (both runs, so the history has two rows)
    let runs = [cold, warm];
    write_dashboard_data(&runs, engine.compiled_shape_count())?;

    // ------------------------------------------------------------
    // SESSION STATISTICS
    // ------------------------------------------------------------

    println!();
    println!("============================================================");
    println!("                  SESSION STATISTICS");
    println!("============================================================");

    println!(
        "Distinct query shapes compiled: {}",
        engine.compiled_shape_count()
    );

    println!("============================================================");

    Ok(())
}

/// Program entry point.
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,

        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
