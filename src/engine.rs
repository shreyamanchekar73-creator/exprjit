use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use arrow::array::{Float64Array, Int64Array};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;

use crate::ast::{BinOp, Expr};
use crate::jit::{ColType, CompiledFilterFn, Jit, Schema};
use crate::parser::parse;

pub struct Engine {
    jit: Jit,
    /// Cranelift's `JITModule` refuses to define two functions with the
    /// same name (`declare_function` + `define_function` panics with
    /// `DuplicateDefinition`), and a single `Engine`/`Jit` is meant to be
    /// reused across many compiles. This counter guarantees every
    /// newly-compiled (i.e. cache-miss) query gets a distinct symbol name.
    next_query_id: u64,
    /// Query-shape -> compiled function. See the module doc comment above.
    cache: HashMap<String, CompiledFilterFn>,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        Engine { jit: Jit::new(), next_query_id: 0, cache: HashMap::new() }
    }

    /// Number of distinct query shapes actually compiled so far (i.e. cache
    /// misses). Useful for demonstrating that repeated structurally-equal
    /// queries don't grow this number.
    pub fn compiled_shape_count(&self) -> usize {
        self.cache.len()
    }

    /// Parse `query` and JIT-compile it (or reuse a cached compilation) against
    /// `batch`'s schema.
    pub fn compile_query(&mut self, query: &str, batch: &RecordBatch) -> Result<CompiledQuery> {
        let expr = parse(query).map_err(|e| anyhow!(e.to_string())).context("parsing query")?;
        self.compile_expr(&expr, batch)
    }

    /// Same as `compile_query`, but skips parsing for a caller who already
    /// has an `Expr` (e.g. built programmatically, as in the benchmarks).
    pub fn compile_expr(&mut self, expr: &Expr, batch: &RecordBatch) -> Result<CompiledQuery> {
        let schema = schema_from_batch(batch)?;

        // FIX 1: validate before handing the AST to Cranelift, so bad queries
        // return a normal error instead of panicking inside jit.rs.
        if check_expr(expr, &schema)? != Kind::Bool {
            bail!("query must be a boolean expression, e.g. `age > 18`");
        }

        let schema_key = schema_cache_key(&schema);
        let key = format!("{}|{}", schema_key, expr_cache_key(expr));

        if let Some(&compiled) = self.cache.get(&key) {
            return Ok(CompiledQuery { compiled, compile_time: Duration::ZERO, cache_hit: true, schema_key });
        }

        let started = Instant::now();
        let fn_name = format!("query_pred_{}", self.next_query_id);
        self.next_query_id += 1;
        let compiled = self.jit.compile(expr, &schema, &fn_name);
        let compile_time = started.elapsed();

        self.cache.insert(key, compiled);
        Ok(CompiledQuery { compiled, compile_time, cache_hit: false, schema_key })
    }
}

pub struct CompiledQuery {
    compiled: CompiledFilterFn,
    /// Wall-clock time spent in Cranelift compilation for this call.
    /// `Duration::ZERO` when `cache_hit` is true (no compilation happened).
    pub compile_time: Duration,
    /// Whether this call reused a previously-compiled function for the same
    /// query shape, rather than invoking Cranelift again.
    pub cache_hit: bool,
    /// FIX 3: the column layout this function was compiled for. `filter`
    /// refuses to run against a batch with a different layout.
    schema_key: String,
}

impl CompiledQuery {
    /// Run the compiled predicate over every row of `batch`, returning the
    /// (0-based) indices of rows that matched.
    pub fn filter(&self, batch: &RecordBatch) -> Result<Vec<u32>> {
        let batch_key = schema_cache_key(&schema_from_batch(batch)?);
        if batch_key != self.schema_key {
            bail!(
                "this query was compiled for a different column layout \
                 (compiled for [{}], got [{}]); recompile it against this batch",
                self.schema_key,
                batch_key
            );
        }
        let col_ptrs = column_pointers(batch)?;
        let mut kept = Vec::new();
        for row in 0..batch.num_rows() {
            // SAFETY: `col_ptrs[i]` points at the start of a contiguous,
            // correctly-typed buffer of at least `batch.num_rows()`
            // elements for every column referenced by `schema` (enforced
            // by `column_pointers` walking every column in the batch,
            // which is exactly what `schema_from_batch` indexed against),
            // and `row` is in `0..batch.num_rows()`.
            let keep = unsafe { (self.compiled)(col_ptrs.as_ptr(), row as i64) };
            if keep != 0 {
                kept.push(row as u32);
            }
        }
        Ok(kept)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Num,
    Bool,
}

/// Type-check an expression against the schema. Returns an error for every
/// case that `jit.rs` would otherwise panic on.
fn check_expr(expr: &Expr, schema: &Schema) -> Result<Kind> {
    Ok(match expr {
        Expr::Column(name) => {
            if !schema.columns.contains_key(name) {
                bail!("unknown column `{name}`");
            }
            Kind::Num
        }
        Expr::IntLit(_) | Expr::FloatLit(_) => Kind::Num,
        Expr::StrLit(_) => bail!("string literals are not supported by the JIT path (only Int64/Float64 columns)"),
        Expr::Unary(_, inner) => {
            if check_expr(inner, schema)? != Kind::Bool {
                bail!("`!` must be applied to a true/false expression, e.g. `!(age > 5)`");
            }
            Kind::Bool
        }
        Expr::Binary(lhs, op, rhs) => {
            let l = check_expr(lhs, schema)?;
            let r = check_expr(rhs, schema)?;
            let want = if matches!(op, BinOp::And | BinOp::Or) { Kind::Bool } else { Kind::Num };
            if l != want || r != want {
                if want == Kind::Bool {
                    bail!("`&&` / `||` need true/false expressions on both sides");
                } else {
                    bail!("comparison {op:?} needs a column or number on both sides");
                }
            }
            Kind::Bool
        }
    })
}

/// Canonical string encoding of an `Expr`'s exact shape -- used as (part of)
/// the compilation-cache key. Two expressions produce the same key if and
/// only if they have the same structure, operators, literals, and column
/// references; this is deliberately independent of the original query
/// string, so `"a>0"` and `"a > 0"` (or any other re-parse of an
/// equivalent expression) hit the same cache entry.
fn expr_cache_key(expr: &Expr) -> String {
    match expr {
        Expr::Column(name) => format!("col({name})"),
        Expr::IntLit(n) => format!("int({n})"),
        // f64's Display is deterministic for a given bit pattern, which is
        // all a cache key needs (it doesn't need to be a valid numeric
        // ordering key, just a stable, unique-enough string).
        Expr::FloatLit(f) => format!("float({f})"),
        Expr::StrLit(s) => format!("str({s:?})"),
        Expr::Unary(op, inner) => format!("un({op:?},{})", expr_cache_key(inner)),
        Expr::Binary(l, op, r) => format!("bin({:?},{},{})", op, expr_cache_key(l), expr_cache_key(r)),
    }
}

/// Canonical string encoding of a `Schema`'s column layout (name, index,
/// type), sorted by name for determinism -- used as (part of) the
/// compilation-cache key, so a compiled function is only reused against a
/// batch with the same column layout it was compiled for.
fn schema_cache_key(schema: &Schema) -> String {
    let mut entries: Vec<(&String, &(usize, ColType))> = schema.columns.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    entries
        .iter()
        .map(|(name, (idx, ty))| format!("{name}:{idx}:{ty:?}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn schema_from_batch(batch: &RecordBatch) -> Result<Schema> {
    let mut schema = Schema::default();
    for (idx, field) in batch.schema().fields().iter().enumerate() {
        let ty = match field.data_type() {
            DataType::Int64 => ColType::Int64,
            DataType::Float64 => ColType::Float64,
            other => bail!(
                "column `{}` has type {other:?}, which the JIT filter path doesn't support \
                 (only Int64/Float64 are lowered right now) \u{2014} use interp::eval_bool for this query instead",
                field.name()
            ),
        };
        schema.columns.insert(field.name().clone(), (idx, ty));
    }
    Ok(schema)
}

/// Build the `*const *const u8` column-pointer array the compiled function
/// expects, one raw pointer per column, in schema order.
fn column_pointers(batch: &RecordBatch) -> Result<Vec<*const u8>> {
    let mut ptrs = Vec::with_capacity(batch.num_columns());
    for (idx, col) in batch.columns().iter().enumerate() {
        if col.null_count() > 0 {
            let name = batch.schema().field(idx).name().clone();
            bail!("column `{name}` contains nulls, which the JIT filter path doesn't handle");
        }
        let ptr = match col.data_type() {
            DataType::Int64 => {
                let arr = col
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| anyhow!("column {idx} claims Int64 but downcast failed"))?;
                arr.values().as_ptr() as *const u8
            }
            DataType::Float64 => {
                let arr = col
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| anyhow!("column {idx} claims Float64 but downcast failed"))?;
                arr.values().as_ptr() as *const u8
            }
            other => bail!("unsupported column type {other:?} at index {idx}"),
        };
        ptrs.push(ptr);
    }
    Ok(ptrs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{Field, Schema as ArrowSchema};
    use std::sync::Arc;

    fn sample() -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("age", DataType::Int64, false),
            Field::new("score", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn unknown_column_is_an_error_not_a_panic() {
        let mut e = Engine::new();
        assert!(e.compile_query("bonus > 5", &sample()).is_err());
    }

    #[test]
    fn string_literal_is_an_error_not_a_panic() {
        let mut e = Engine::new();
        assert!(e.compile_query(r#"age == "x""#, &sample()).is_err());
    }

    #[test]
    fn non_boolean_query_is_an_error() {
        let mut e = Engine::new();
        assert!(e.compile_query("age", &sample()).is_err());
    }

    #[test]
    fn boolean_inside_comparison_is_an_error() {
        let mut e = Engine::new();
        assert!(e.compile_query("(age > 1) > 2", &sample()).is_err());
        assert!(e.compile_query("age && score", &sample()).is_err());
    }

    #[test]
    fn repeated_query_hits_cache() {
        let mut e = Engine::new();
        let b = sample();
        assert!(!e.compile_query("age > 1", &b).unwrap().cache_hit);
        assert!(e.compile_query("age > 1", &b).unwrap().cache_hit);
        assert_eq!(e.compiled_shape_count(), 1);
    }

    #[test]
    fn filter_rejects_batch_with_different_layout() {
        let mut e = Engine::new();
        let q = e.compile_query("age > 1", &sample()).unwrap();
        // same columns, different order -> different layout
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("score", DataType::Float64, false),
            Field::new("age", DataType::Int64, false),
        ]));
        let other = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Int64Array::from(vec![1, 2, 3])),
            ],
        )
        .unwrap();
        assert!(q.filter(&other).is_err());
        assert_eq!(q.filter(&sample()).unwrap(), vec![1, 2]);
    }
}
