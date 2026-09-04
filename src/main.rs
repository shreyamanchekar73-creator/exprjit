//! ExprJIT CLI.
//!
//! Usage:
//!   exprjit                                  Run a demo query on a small
//!                                             built-in sample batch.
//!   exprjit "<query>"                        Run a custom query on the
//!                                             built-in sample batch.
//!   exprjit --csv <path> "<query>"           Load a CSV file and run a
//!                                             query against it.
//!   exprjit --csv <path> --list-columns      Print the CSV's inferred
//!                                             schema (name + type) and exit.
//!   exprjit --help                           Show this usage text.
//!
//! CSV format: first line is a comma-separated header of column names;
//! every column must be entirely Int64 or entirely Float64 (see
//! `csv_loader.rs` for exact rules). No quoted fields.

use std::env;
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;

use exprjit::csv_loader::load_csv;
use exprjit::engine::Engine;

const USAGE: &str = r#"exprjit - JIT-compiled query filtering over Arrow record batches

USAGE:
    exprjit                                Run a demo query on a built-in sample batch
    exprjit "<query>"                      Run a custom query on the built-in sample batch
    exprjit --csv <path> "<query>"         Load a CSV file and run a query against it
    exprjit --csv <path> --list-columns    Print the CSV's inferred schema and exit
    exprjit --help                         Show this message

QUERY SYNTAX:
    Comparisons: >  <  >=  <=  ==  !=
    Logic:       &&  ||  !
    Values:      column names, integer/float literals, "quoted strings"
    Grouping:    ( ... )

    Example: "age > 18 && score >= 4.5"

CSV FORMAT:
    First line is a comma-separated header of column names. Every column
    must be entirely Int64 or entirely Float64 -- see csv_loader.rs for the
    exact rules. Quoted fields / embedded commas are not supported.
"#;

/// A small hardcoded batch to demo against when no --csv is given: 20 rows
/// of `age` (Int64) and `score` (Float64).
fn sample_batch() -> Result<RecordBatch> {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("age", DataType::Int64, false),
        Field::new("score", DataType::Float64, false),
    ]));
    let age = Int64Array::from((0..20i64).collect::<Vec<_>>());
    let score = Float64Array::from((0..20).map(|i| i as f64 * 1.3).collect::<Vec<_>>());
    RecordBatch::try_new(schema, vec![Arc::new(age), Arc::new(score)]).context("building sample batch")
}

fn list_columns(batch: &RecordBatch) {
    println!("columns:");
    for field in batch.schema().fields() {
        println!("  {:<20} {:?}", field.name(), field.data_type());
    }
    println!("rows: {}", batch.num_rows());
}

/// Hand-rolled arg parsing (no `clap`): this project already pulls in
/// `criterion` for benchmarking, whose dependency tree needs a newer Cargo
/// than some environments have available; adding another crate with the
/// same requirement for CLI parsing isn't worth it for three flags.
struct Args {
    csv_path: Option<String>,
    query: Option<String>,
    list_columns: bool,
    help: bool,
}

fn parse_args() -> Args {
    let mut args = Args { csv_path: None, query: None, list_columns: false, help: false };
    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--csv" => args.csv_path = it.next(),
            "--list-columns" => args.list_columns = true,
            "--help" | "-h" => args.help = true,
            other => args.query = Some(other.to_string()),
        }
    }
    args
}

fn run() -> Result<()> {
    let args = parse_args();

    if args.help {
        print!("{USAGE}");
        return Ok(());
    }

    let batch = match &args.csv_path {
        Some(path) => load_csv(path).with_context(|| format!("loading CSV from {path}"))?,
        None => sample_batch()?,
    };

    if args.list_columns {
        list_columns(&batch);
        return Ok(());
    }

    let query = args.query.unwrap_or_else(|| "age > 10 && score < 20.0".to_string());

    let mut engine = Engine::new();
    let compiled = engine.compile_query(&query, &batch).with_context(|| format!("compiling query `{query}`"))?;
    let kept = compiled.filter(&batch).with_context(|| format!("running query `{query}`"))?;

    println!("query:   {query}");
    println!("matched: {} of {} rows", kept.len(), batch.num_rows());
    println!("indices: {kept:?}");
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
