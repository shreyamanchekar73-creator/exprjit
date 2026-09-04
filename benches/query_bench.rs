//! Criterion benchmarks for the JIT-compiled query predicates.
//!
//! `Cargo.toml` has `harness = false` for this bench target, which means
//! Criterion supplies `fn main()` itself via `criterion_main!` at the
//! bottom of this file. If that macro invocation is ever removed (or ends
//! up nested inside another item instead of at top level), `cargo bench`
//! fails with E0601 ("main function not found") because nothing else in
//! this file generates an entry point for the `query_bench` binary.

use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use exprjit::ast::{BinOp, Expr};
use exprjit::engine::Engine;

fn build_batch(rows: usize) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Float64, false),
    ]));
    let col_a = Int64Array::from((0..rows as i64).collect::<Vec<_>>());
    let col_b = Float64Array::from((0..rows).map(|i| i as f64 * 0.5).collect::<Vec<_>>());
    RecordBatch::try_new(schema, vec![Arc::new(col_a), Arc::new(col_b)]).expect("building bench batch")
}

/// (a > 10) && (b < 5.0), built directly as an `Expr` rather than parsed,
/// so this benchmark measures JIT + execution, not the parser.
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

fn bench_jit_filter(c: &mut Criterion) {
    let batch = build_batch(10_000);
    let expr = build_expr();

    let mut engine = Engine::new();
    let compiled = engine.compile_expr(&expr, &batch).expect("compiling bench query");

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
            black_box(engine.compile_expr(&expr, &batch).expect("compiling bench query"));
        })
    });
}

criterion_group!(benches, bench_jit_filter, bench_compile_only);
criterion_main!(benches);
