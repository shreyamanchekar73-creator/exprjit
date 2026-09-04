//! Glue layer between a parsed query, the JIT, and an Arrow `RecordBatch`.
//!
//! `Engine::compile_query` parses + JIT-compiles a query string against a
//! batch's schema; the resulting `CompiledQuery::filter` runs the compiled
//! predicate over every row and returns the indices that matched.
//!
//! Only `Int64` and `Float64` columns are supported, matching what
//! `jit.rs::ColType` and `ExprCodegen::gen_column_load` know how to load.
//! Anything else (strings, nulls, other numeric widths) is rejected up
//! front with a clear error rather than silently producing wrong results;
//! route those queries through `interp.rs` instead.

use anyhow::{anyhow, bail, Context, Result};
use arrow::array::{Float64Array, Int64Array};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;

use crate::ast::Expr;
use crate::jit::{ColType, CompiledFilterFn, Jit, Schema};
use crate::parser::parse;

pub struct Engine {
    jit: Jit,
    /// Cranelift's `JITModule` refuses to define two functions with the
    /// same name (`declare_function` + `define_function` panics with
    /// `DuplicateDefinition`), and a single `Engine`/`Jit` is meant to be
    /// reused across many `compile_query`/`compile_expr` calls (see e.g.
    /// `tests/jit_vs_interp.rs`, which compiles ~18 queries against one
    /// `Engine`). This counter guarantees every compiled query gets a
    /// distinct symbol name.
    next_query_id: u64,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        Engine { jit: Jit::new(), next_query_id: 0 }
    }

    /// Parse `query` and JIT-compile it against `batch`'s schema.
    pub fn compile_query(&mut self, query: &str, batch: &RecordBatch) -> Result<CompiledQuery> {
        let expr = parse(query).map_err(|e| anyhow!(e.to_string())).context("parsing query")?;
        self.compile_expr(&expr, batch)
    }

    /// Same as `compile_query`, but skips parsing for a caller who already
    /// has an `Expr` (e.g. built programmatically, as in the benchmarks).
    pub fn compile_expr(&mut self, expr: &Expr, batch: &RecordBatch) -> Result<CompiledQuery> {
        let schema = schema_from_batch(batch)?;
        let fn_name = format!("query_pred_{}", self.next_query_id);
        self.next_query_id += 1;
        let compiled = self.jit.compile(expr, &schema, &fn_name);
        Ok(CompiledQuery { compiled })
    }
}

pub struct CompiledQuery {
    compiled: CompiledFilterFn,
}

impl CompiledQuery {
    /// Run the compiled predicate over every row of `batch`, returning the
    /// (0-based) indices of rows that matched.
    pub fn filter(&self, batch: &RecordBatch) -> Result<Vec<u32>> {
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
