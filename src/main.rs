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

const USAGE: &str = "\
ExprJIT — expression JIT query engine

Usage:
  exprjit [--csv FILE] [--list-columns] [QUERY]
  exprjit --server

Options:
  --csv FILE       Load data from a CSV file
  --list-columns   List columns from the CSV file
  --server         Start the dashboard API server
  -h, --help       Show this help message

Examples:
  exprjit \"age > 30\"
  exprjit --csv people.csv \"age > 30\"
  exprjit --csv people.csv --list-columns
  exprjit --server
";

#[derive(Debug, Default)]
struct Args {
    csv_path: Option<String>,
    list_columns: bool,
    server: bool,
    query: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut args = Args::default();
    let mut it = env::args().skip(1);

    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }

            "--csv" => match it.next() {
                Some(path) if !path.starts_with("--") => {
                    args.csv_path = Some(path);
                }
                _ => {
                    anyhow::bail!(
                        "--csv needs a file path, e.g. --csv data.csv"
                    );
                }
            },

            "--list-columns" => {
                args.list_columns = true;
            }

            "--server" => {
                args.server = true;
            }

            _ if arg.starts_with("--") => {
                anyhow::bail!("Unknown option: {arg}\n\n{USAGE}");
            }

            _ => {
                if args.query.is_some() {
                    anyhow::bail!(
                        "Only one query expression is supported.\n\n{USAGE}"
                    );
                }
                args.query = Some(arg);
            }
        }
    }

    Ok(args)
}

fn sample_batch(rows: usize) -> Result<RecordBatch> {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("age", DataType::Int64, false),
        Field::new("score", DataType::Float64, false),
    ]));

    let age = Int64Array::from((0..rows as i64).collect::<Vec<_>>());

    let score = Float64Array::from(
        (0..rows)
            .map(|i| i as f64 * 1.3)
            .collect::<Vec<_>>(),
    );

    RecordBatch::try_new(
        schema,
        vec![Arc::new(age), Arc::new(score)],
    )
    .context("building sample data")
}

fn interpreter_filter(
    batch: &RecordBatch,
    expr: &Expr,
) -> Result<Vec<u32>> {
    let schema = batch.schema();

    enum Column<'a> {
        Int(&'a Int64Array),
        Float(&'a Float64Array),
    }

    let mut columns = Vec::new();

    for (i, field) in schema.fields().iter().enumerate() {
        let column = match batch.column(i).data_type() {
            DataType::Int64 => Column::Int(
                batch
                    .column(i)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .context("reading Int64 column")?,
            ),

            DataType::Float64 => Column::Float(
                batch
                    .column(i)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .context("reading Float64 column")?,
            ),

            other => {
                anyhow::bail!(
                    "Unsupported column type for interpreter: {other:?}"
                );
            }
        };

        columns.push((field.name().clone(), column));
    }

    let mut result = Vec::new();

    let mut row: HashMap<&str, Value> = HashMap::new();

    for row_idx in 0..batch.num_rows() {
        row.clear();

        for (name, column) in &columns {
            let value = match column {
                Column::Int(array) => Value::Int(array.value(row_idx)),
                Column::Float(array) => Value::Float(array.value(row_idx)),
            };

            row.insert(name.as_str(), value);
        }

        if interp::eval_bool(expr, &row)
            .with_context(|| format!("interpreting row {row_idx}"))?
        {
            result.push(row_idx as u32);
        }
    }

    Ok(result)
}

#[derive(Debug, Clone)]
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

fn write_dashboard_data(stats: &[QueryStats]) -> Result<()> {
    let latest = stats.last();

    let current_query = latest
        .map(|s| s.query.clone())
        .unwrap_or_default();

    let total_rows = latest
        .map(|s| s.total_rows)
        .unwrap_or(0);

    let matched_rows = latest
        .map(|s| s.matched_rows)
        .unwrap_or(0);

    let match_pct = latest
        .map(|s| s.match_pct)
        .unwrap_or(0.0);

    let compile_us = latest
        .map(|s| s.compile_us)
        .unwrap_or(0.0);

    let jit_us = latest
        .map(|s| s.jit_us)
        .unwrap_or(0.0);

    let interpreter_us = latest
        .map(|s| s.interpreter_us)
        .unwrap_or(0.0);

    let speedup = latest
        .map(|s| s.speedup)
        .unwrap_or(0.0);

    let cache_status = latest
        .map(|s| s.cache_status.clone())
        .unwrap_or_else(|| "N/A".to_string());

    let correctness = latest
        .map(|s| s.correctness.clone())
        .unwrap_or_else(|| "N/A".to_string());

    let indices = latest
        .map(|s| s.indices.clone())
        .unwrap_or_default();

    let history = stats
        .iter()
        .map(|s| {
            format!(
                r#"{{"query":"{}","matched_rows":{},"jit_us":{},"interpreter_us":{},"speedup":{},"cache_status":"{}","correctness":"{}"}}"#,
                escape_json(&s.query),
                s.matched_rows,
                s.jit_us,
                s.interpreter_us,
                s.speedup,
                escape_json(&s.cache_status),
                escape_json(&s.correctness),
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    let indices_json = indices
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");

    let js = format!(
        r#"window.EXPRJIT_DATA = {{
  "query": "{}",
  "total_rows": {},
  "matched_rows": {},
  "match_pct": {},
  "compile_us": {},
  "jit_us": {},
  "interpreter_us": {},
  "speedup": {},
  "cache_status": "{}",
  "correctness": "{}",
  "indices": [{}],
  "history": [{}]
}};
"#,
        escape_json(&current_query),
        total_rows,
        matched_rows,
        match_pct,
        compile_us,
        jit_us,
        interpreter_us,
        speedup,
        escape_json(&cache_status),
        escape_json(&correctness),
        indices_json,
        history,
    );

    fs::write("dashboard_data.js", js)
        .context("writing dashboard_data.js")?;

    Ok(())
}

fn escape_json(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn run_query(
    engine: &mut Engine,
    batch: &RecordBatch,
    query: &str,
    label: &str,
) -> Result<QueryStats> {
    println!("\n[{label}] Query: {query}");

    let compiled = engine
        .compile_query(query, batch)
        .with_context(|| format!("compiling query `{query}`"))?;

    let compile_us =
        compiled.compile_time.as_secs_f64() * 1_000_000.0;

    let cache_status = if compiled.cache_hit {
        "HIT"
    } else {
        "MISS"
    };

    let started = Instant::now();

    let indices = compiled
        .filter(batch)
        .with_context(|| format!("executing query `{query}`"))?;

    let jit_time = started.elapsed();

    let expr = parse(query)
        .map_err(|e| anyhow::anyhow!(e.to_string()))
        .context("parsing query for interpreter comparison")?;

    let started = Instant::now();

    let interpreter_indices = interpreter_filter(batch, &expr)?;

    let interpreter_time = started.elapsed();

    let speedup = if jit_time.as_nanos() > 0 {
        interpreter_time.as_secs_f64()
            / jit_time.as_secs_f64()
    } else {
        0.0
    };

    let match_pct = if batch.num_rows() > 0 {
        indices.len() as f64 * 100.0
            / batch.num_rows() as f64
    } else {
        0.0
    };

    let correctness = if indices == interpreter_indices {
        "PASS"
    } else {
        "FAIL"
    };

    println!("  Total rows:      {}", batch.num_rows());
    println!("  Matched rows:    {}", indices.len());
    println!("  Match percentage:{match_pct:>8.2}%");
    println!("  Compile time:    {compile_us:.2} us");
    println!(
        "  JIT execution:   {:.2} us",
        jit_time.as_secs_f64() * 1_000_000.0
    );
    println!(
        "  Interpreter:     {:.2} us",
        interpreter_time.as_secs_f64() * 1_000_000.0
    );
    println!("  Speedup:         {speedup:.2}x");
    println!("  Cache:           {cache_status}");
    println!("  Correctness:     {correctness}");
    println!("  Matching indices:{indices:?}");

    Ok(QueryStats {
        query: query.to_string(),
        total_rows: batch.num_rows(),
        matched_rows: indices.len(),
        match_pct,
        compile_us,
        jit_us: jit_time.as_secs_f64() * 1_000_000.0,
        interpreter_us: interpreter_time.as_secs_f64() * 1_000_000.0,
        speedup,
        cache_status: cache_status.to_string(),
        correctness: correctness.to_string(),
        indices,
    })
}

fn run() -> Result<()> {
    let args = parse_args()?;

    // Start the API server when --server is supplied.
    if args.server {
        return exprjit::api::serve();
    }

    let batch = if let Some(csv_path) = &args.csv_path {
        load_csv(csv_path)
            .with_context(|| format!("loading CSV file `{csv_path}`"))?
    } else {
        sample_batch(20)?
    };

    if args.list_columns {
        println!("Columns:");

        for field in batch.schema().fields() {
            println!(
                "  {} ({:?})",
                field.name(),
                field.data_type()
            );
        }

        return Ok(());
    }

    let query = args
        .query
        .unwrap_or_else(|| "age > 10".to_string());

    let mut engine = Engine::new();
    let mut history = Vec::new();

    let cold_stats = run_query(
        &mut engine,
        &batch,
        &query,
        "COLD RUN",
    )?;

    history.push(cold_stats);

    let warm_stats = run_query(
        &mut engine,
        &batch,
        &query,
        "WARM RUN",
    )?;

    history.push(warm_stats);

    write_dashboard_data(&history)?;

    println!("\nDashboard data written to dashboard_data.js");

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,

        Err(error) => {
            eprintln!("Error: {error:#}");
            ExitCode::FAILURE
        }
    }
}