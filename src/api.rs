use axum::http::{header, Method};
use tower_http::cors::{Any, CorsLayer};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use anyhow::{Context, Result};
use arrow::array::{Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use axum::{
    extract::State,
    http::StatusCode,
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::engine::Engine;
use crate::interp::{self, Value};
use crate::parser::parse;

#[derive(Deserialize)]
pub struct QueryRequest {
    pub query: String,
}

#[derive(Serialize)]
pub struct QueryResponse {
    pub query: String,
    pub total_rows: usize,
    pub matched_rows: usize,
    pub match_pct: f64,
    pub indices: Vec<u32>,
    pub compile_us: f64,
    pub jit_us: f64,
    pub interpreter_us: f64,
    pub speedup: f64,
    pub cache_status: String,
    pub correctness: String,
}

struct WorkerRequest {
    query: String,
    reply: oneshot::Sender<Result<QueryResponse, String>>,
}

type AppState = mpsc::Sender<WorkerRequest>;

fn sample_batch(rows: usize) -> Result<RecordBatch> {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("age", DataType::Int64, false),
        Field::new("score", DataType::Float64, false),
    ]));

    let age = Int64Array::from((0..rows as i64).collect::<Vec<_>>());
    let score =
        Float64Array::from((0..rows).map(|i| i as f64 * 1.3).collect::<Vec<_>>());

    RecordBatch::try_new(schema, vec![Arc::new(age), Arc::new(score)])
        .context("building sample data")
}

fn execute_query(
    engine: &mut Engine,
    batch: &RecordBatch,
    query: &str,
) -> Result<QueryResponse> {
    let compiled = engine
        .compile_query(query, batch)
        .with_context(|| format!("compiling query `{query}`"))?;

    let compile_us = compiled.compile_time.as_secs_f64() * 1_000_000.0;
    let cache_status = if compiled.cache_hit { "HIT" } else { "MISS" };

    let started = Instant::now();
    let indices = compiled
        .filter(batch)
        .with_context(|| format!("executing query `{query}`"))?;
    let jit_time = started.elapsed();

    let expr = parse(query)
        .map_err(|e| anyhow::anyhow!(e.to_string()))
        .context("parsing query for interpreter comparison")?;

    let schema = batch.schema();

    enum Column<'a> {
        Int(&'a Int64Array),
        Float(&'a Float64Array),
    }

    let mut columns = Vec::new();

    for (i, field) in schema.fields().iter().enumerate() {
        let column = match batch.column(i).data_type() {
            DataType::Int64 => Column::Int(
                batch.column(i)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .context("reading Int64 column")?,
            ),
            DataType::Float64 => Column::Float(
                batch.column(i)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .context("reading Float64 column")?,
            ),
            other => anyhow::bail!("unsupported column type: {other:?}"),
        };

        columns.push((field.name().clone(), column));
    }

    let started = Instant::now();
    let mut interpreter_indices = Vec::new();
    let mut row: std::collections::HashMap<&str, Value> =
        std::collections::HashMap::new();

    for row_idx in 0..batch.num_rows() {
        row.clear();

        for (name, column) in &columns {
            let value = match column {
                Column::Int(array) => Value::Int(array.value(row_idx)),
                Column::Float(array) => Value::Float(array.value(row_idx)),
            };

            row.insert(name.as_str(), value);
        }

        if interp::eval_bool(&expr, &row)
            .with_context(|| format!("interpreting row {row_idx}"))?
        {
            interpreter_indices.push(row_idx as u32);
        }
    }

    let interpreter_time = started.elapsed();

    let speedup = if jit_time.as_nanos() > 0 {
        interpreter_time.as_secs_f64() / jit_time.as_secs_f64()
    } else {
        0.0
    };

    let match_pct = if batch.num_rows() > 0 {
        indices.len() as f64 * 100.0 / batch.num_rows() as f64
    } else {
        0.0
    };

    Ok(QueryResponse {
        query: query.to_string(),
        total_rows: batch.num_rows(),
        matched_rows: indices.len(),
        match_pct,
        correctness: if indices == interpreter_indices {
            "PASS".to_string()
        } else {
            "FAIL".to_string()
        },
        indices,
        compile_us,
        jit_us: jit_time.as_secs_f64() * 1_000_000.0,
        interpreter_us: interpreter_time.as_secs_f64() * 1_000_000.0,
        speedup,
        cache_status: cache_status.to_string(),
    })
}

async fn query_handler(
    State(sender): State<AppState>,
    Json(request): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, (StatusCode, String)> {
    let (reply, receiver) = oneshot::channel();

    sender
        .send(WorkerRequest {
            query: request.query,
            reply,
        })
        .map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Query worker is unavailable".to_string(),
            )
        })?;

    let result = receiver.await.map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Query worker stopped unexpectedly".to_string(),
        )
    })?;

    result
        .map(Json)
        .map_err(|error| (StatusCode::BAD_REQUEST, error))
}

pub fn serve() -> Result<()> {
    let (sender, receiver) = mpsc::channel::<WorkerRequest>();

    thread::spawn(move || {
        let mut engine = Engine::new();

        let batch = match sample_batch(20) {
            Ok(batch) => batch,
            Err(error) => {
                eprintln!("Could not create sample data: {error:#}");
                return;
            }
        };

        while let Ok(request) = receiver.recv() {
            let result = execute_query(&mut engine, &batch, &request.query)
                .map_err(|error| format!("{error:#}"));

            let _ = request.reply.send(result);
        }
    });

    let app = Router::new()
    .route("/api/query", post(query_handler))
    .layer(
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods([Method::POST, Method::OPTIONS])
            .allow_headers([header::CONTENT_TYPE]),
    )
    .with_state(sender);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("creating Tokio runtime")?;

    runtime.block_on(async {
        let port = std::env::var("PORT")
    .unwrap_or_else(|_| "3000".to_string());

let addr = format!("0.0.0.0:{}", port);

let listener = tokio::net::TcpListener::bind(&addr)
    .await
    .context("binding API listener")?;

println!("ExprJIT API running at http://{}", addr);
println!("Query endpoint: POST http://{}/api/query", addr);
        axum::serve(listener, app)
            .await
            .context("running API server")
    })
}