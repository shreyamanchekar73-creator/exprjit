//! Cross-checks the JIT-compiled predicate path (engine.rs -> jit.rs) against
//! the tree-walking reference interpreter (interp.rs) across a spread of
//! queries and rows, including edge cases (negatives, zero, values sitting
//! right on comparison boundaries). If these two paths ever disagree, it
//! means the JIT codegen (most likely `ExprCodegen::gen_binop`'s icmp/fcmp
//! handling — see the version note at the top of jit.rs) has drifted from
//! the interpreter's semantics, which are the source of truth here.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;

use exprjit::engine::Engine;
use exprjit::interp::{eval_bool, Row, Value};
use exprjit::parser::parse;

/// Two Int64/Float64 columns with deliberately awkward values: negatives,
/// zero, and numbers sitting exactly on the thresholds the QUERIES below
/// compare against, so `>` vs `>=` and `<` vs `<=` are actually exercised
/// at their boundaries rather than just in the "obviously true/false"
/// middle of the range.
fn build_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Float64, false),
    ]));
    let a_vals: Vec<i64> = vec![-5, -1, 0, 1, 5, 10, 10, 11, 50, 99, -100, 3, 7, 20, 0];
    let b_vals: Vec<f64> = vec![
        -2.5, 0.0, 5.0, 4.999, 5.001, 100.0, -3.3, 2.2, 8.8, 5.0, 1.0, 9.9, 0.1, -0.1, 5.5,
    ];
    assert_eq!(a_vals.len(), b_vals.len(), "test fixture columns must be the same length");
    let col_a = Int64Array::from(a_vals);
    let col_b = Float64Array::from(b_vals);
    RecordBatch::try_new(schema, vec![Arc::new(col_a), Arc::new(col_b)]).expect("building test batch")
}

/// Pull row `row_idx` out of `batch` as owned (name, Value) pairs. Kept
/// separate from `Row` construction (which needs `&str` keys) so the caller
/// controls how long the owned Strings live relative to the borrowed map.
fn row_values(batch: &RecordBatch, row_idx: usize) -> Vec<(String, Value)> {
    let schema = batch.schema();
    let mut out = Vec::with_capacity(schema.fields().len());
    for (col_idx, field) in schema.fields().iter().enumerate() {
        let col = &batch.columns()[col_idx];
        let value = match field.data_type() {
            DataType::Int64 => Value::Int(
                col.as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("column typed Int64 in schema but downcast failed")
                    .values()[row_idx],
            ),
            DataType::Float64 => Value::Float(
                col.as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("column typed Float64 in schema but downcast failed")
                    .values()[row_idx],
            ),
            other => panic!("test fixture has unsupported column type {other:?}"),
        };
        out.push((field.name().clone(), value));
    }
    out
}

/// Every comparison/logical operator, both orderings of And/Or, negation,
/// parens, and values that land exactly on a boundary (e.g. `a == 0`,
/// `b == 5.0`) so equality isn't only tested against values that never
/// actually match.
const QUERIES: &[&str] = &[
    "a > 0",
    "a >= 10",
    "a < 0",
    "a <= 0",
    "a == 0",
    "a != 0",
    "b > 5.0",
    "b >= 5.0",
    "b < 0.0",
    "b <= 5.0",
    "b == 5.0",
    "b != 5.0",
    "a > 0 && b > 0.0",
    "a > 0 || b < 0.0",
    "!(a > 0)",
    "!(a > 0 && b > 0.0)",
    "a >= 1 && a <= 10",
    "(a > 50 || a < 3) && b != 0.0",
];

#[test]
fn jit_and_interpreter_agree_on_every_row() {
    let batch = build_batch();
    let mut engine = Engine::new();
    let mut mismatches = Vec::new();

    for query in QUERIES {
        let expr = parse(query).unwrap_or_else(|e| panic!("failed to parse `{query}`: {e}"));
        let compiled = engine
            .compile_expr(&expr, &batch)
            .unwrap_or_else(|e| panic!("failed to JIT-compile `{query}`: {e}"));

        let jit_matches: HashSet<u32> = compiled.filter(&batch).expect("jit filter failed").into_iter().collect();

        for row_idx in 0..batch.num_rows() {
            let values = row_values(&batch, row_idx);
            let row: Row = values.iter().map(|(k, v)| (k.as_str(), v.clone())).collect::<HashMap<_, _>>();

            let interp_result =
                eval_bool(&expr, &row).unwrap_or_else(|e| panic!("interpreter failed on `{query}` row {row_idx}: {e}"));
            let jit_result = jit_matches.contains(&(row_idx as u32));

            if interp_result != jit_result {
                mismatches.push(format!(
                    "query `{query}` row {row_idx}: interp={interp_result} jit={jit_result} row={values:?}"
                ));
            }
        }
    }

    assert!(
        mismatches.is_empty(),
        "JIT and interpreter disagreed on {} case(s):\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}
