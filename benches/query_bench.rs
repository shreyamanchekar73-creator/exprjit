use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use exprjit::ast::{BinOp, Expr};
use exprjit::engine::Engine;
use exprjit::interp::{eval_bool, Row, Value};

fn build_batch(rows: usize) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Float64, false),
    ]));
    let col_a = Int64Array::from((0..rows as i64).collect::<Vec<_>>());
    let col_b = Float64Array::from((0..rows).map(|i| i as f64 * 0.5).collect::<Vec<_>>());
    RecordBatch::try_new(schema, vec![Arc::new(col_a), Arc::new(col_b)])
        .expect("building bench batch")
}

/// (a > 10) && (b < 5.0), built directly as an `Expr` rather than parsed,
/// so the filter/compile benchmarks measure JIT + execution, not the parser.
fn build_expr() -> Expr {
    Expr::Binary(
        Box::new(Expr::Binary(
            Box::new(Expr::Column("a".to_string())),
            BinOp::Gt,
            Box::new(Expr::IntLit(10)),
        )),
        BinOp::And,
        Box::new(Expr::Binary(
            Box::new(Expr::Column("b".to_string())),
            BinOp::Lt,
            Box::new(Expr::FloatLit(5.0)),
        )),
    )
}

/// A second, structurally different expression, used only to give the
/// "cold compile" benchmark a query shape it hasn't seen before on each
/// iteration's fresh `Engine` (see `bench_jit_compile_cold`).
fn build_expr_variant(threshold: i64) -> Expr {
    Expr::Binary(
        Box::new(Expr::Column("a".to_string())),
        BinOp::Gt,
        Box::new(Expr::IntLit(threshold)),
    )
}

fn bench_jit_filter(c: &mut Criterion) {
    let batch = build_batch(10_000);
    let expr = build_expr();

    let mut engine = Engine::new();
    let compiled = engine
        .compile_expr(&expr, &batch)
        .expect("compiling bench query");

    c.bench_function("jit_filter_10k_rows", |b| {
        b.iter(|| black_box(compiled.filter(&batch).expect("filter").len()))
    });
}

fn bench_compile_only(c: &mut Criterion) {
    let batch = build_batch(10_000);
    let expr = build_expr();

    c.bench_function("jit_compile_only", |b| {
        b.iter(|| {
            let mut engine = Engine::new();
            black_box(
                engine
                    .compile_expr(&expr, &batch)
                    .expect("compiling bench query"),
            );
        })
    });
}

/// Reference-interpreter equivalent of `bench_jit_filter`: same batch, same
/// logical query, evaluated by walking the AST per row via `interp.rs`
/// instead of running compiled machine code. This is what the JIT is
/// competing against.
fn bench_interp_filter(c: &mut Criterion) {
    let batch = build_batch(10_000);
    let expr = build_expr();

    // interp::eval_bool takes a Row (HashMap<&str, Value>) per call, so
    // extract each row's values up front the same way
    // tests/jit_vs_interp.rs does, to keep the timed closure itself
    // focused on evaluation rather than Arrow downcasting.
    let a_vals = batch.columns()[0]
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec();
    let b_vals = batch.columns()[1]
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .values()
        .to_vec();

    c.bench_function("interp_filter_10k_rows", |b| {
        b.iter(|| {
            let mut kept = 0usize;
            for i in 0..a_vals.len() {
                let mut row: Row = HashMap::new();
                row.insert("a", Value::Int(a_vals[i]));
                row.insert("b", Value::Float(b_vals[i]));
                if eval_bool(&expr, &row).expect("interp eval") {
                    kept += 1;
                }
            }
            black_box(kept)
        })
    });
}

/// Cold path: a fresh `Engine` (empty cache) compiling a query shape it has
/// never seen. This is the baseline "always pay compilation cost" scenario
/// the cache in `engine.rs` is meant to avoid on repeated queries.
fn bench_jit_compile_cold(c: &mut Criterion) {
    let batch = build_batch(10_000);
    let mut counter = 0i64;

    c.bench_function("jit_compile_cold", |b| {
        b.iter(|| {
            // A fresh Engine each iteration, and a different literal each
            // time, so this is always a genuine cache miss -- an honest
            // "first time compiling this exact query" measurement.
            let mut engine = Engine::new();
            counter += 1;
            let expr = build_expr_variant(counter);
            black_box(
                engine
                    .compile_expr(&expr, &batch)
                    .expect("compiling bench query"),
            );
        })
    });
}

/// Warm path: the exact same query shape compiled once up front, then
/// looked up from the cache on every timed iteration. Compare this
/// against `jit_compile_cold` to see the cache's actual benefit.
fn bench_jit_compile_cached(c: &mut Criterion) {
    let batch = build_batch(10_000);
    let expr = build_expr();
    let mut engine = Engine::new();
    // Prime the cache once, outside the timed section.
    engine.compile_expr(&expr, &batch).expect("priming cache");

    c.bench_function("jit_compile_cached", |b| {
        b.iter(|| {
            let compiled = engine.compile_expr(&expr, &batch).expect("cached compile");
            assert!(
                compiled.cache_hit,
                "expected a cache hit in the cached benchmark"
            );
            black_box(compiled);
        })
    });
}

criterion_group!(
    benches,
    bench_jit_filter,
    bench_compile_only,
    bench_interp_filter,
    bench_jit_compile_cold,
    bench_jit_compile_cached
);
criterion_main!(benches);
