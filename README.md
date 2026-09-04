# exprjit

A small **JIT compiler for a query expression language**, built on top of
[Cranelift](https://cranelift.dev/) for native code generation and
[Apache Arrow](https://arrow.apache.org/) for columnar data.

Given a query like `"age > 18 && score >= 4.5"`, exprjit parses it, compiles
it directly to native machine code (not bytecode, not an interpreter loop),
and runs that compiled predicate over every row of an Arrow `RecordBatch` to
produce the matching row indices.

## Why JIT a query language?

A tree-walking interpreter re-parses the AST on every single row. A JIT
compiles the predicate **once** into a tight native function, so filtering
becomes a straight loop of pointer loads and comparisons with no
per-row interpretation overhead. This project includes both:

- `interp.rs` — a straightforward tree-walking reference interpreter
- `jit.rs` — the real thing: compiles an `Expr` to native code via Cranelift

...and a test (`tests/jit_vs_interp.rs`) that proves they agree, so the
interpreter serves as a correctness oracle for the JIT.

## Architecture

```
query string
     |
     v
  lexer.rs   -- hand-rolled tokenizer (no external dependency)
     |
     v
  parser.rs  -- hand-rolled recursive-descent parser -> ast::Expr
     |
     +----------------------------+
     |                            |
     v                            v
  interp.rs                    jit.rs
  (tree-walking                (Cranelift codegen:
   reference eval)               Expr -> native fn ptr)
                                   |
                                   v
                              engine.rs
                              (wires Schema + JITed fn
                               to an Arrow RecordBatch,
                               runs it row-by-row)
```

| File | Role |
|---|---|
| `ast.rs` | The `Expr`/`BinOp`/`UnOp` grammar |
| `lexer.rs` | Dependency-free tokenizer |
| `parser.rs` | Dependency-free recursive-descent parser |
| `interp.rs` | Tree-walking reference evaluator (correctness oracle, string-comparison fallback) |
| `jit.rs` | Cranelift codegen: compiles an `Expr` to a native `unsafe extern "C" fn(*const *const u8, i64) -> u8` |
| `engine.rs` | Glue: builds a `jit::Schema` from an Arrow schema, extracts raw column pointers, runs the compiled predicate over every row |
| `csv_loader.rs` | Minimal dependency-free CSV → `RecordBatch` loader |
| `main.rs` | CLI |

## Building and running

```bash
cargo build
```

Run the built-in demo (20 synthetic rows of `age`/`score`):

```bash
cargo run -- "age > 10 && score < 20.0"
```

Load your own CSV and query it:

```bash
cargo run -- --csv data.csv "age > 30 && salary < 100000.0"
cargo run -- --csv data.csv --list-columns   # inspect inferred schema first
```

CSV format: first line is a header of column names; every column must be
entirely parseable as `Int64` or entirely as `Float64` (see `csv_loader.rs`
doc comments for the exact rules — no quoted fields, no nulls).

## Testing

```bash
cargo test
```

This runs:
- Unit tests for the lexer, parser, interpreter, and CSV loader
- `tests/jit_vs_interp.rs` — an integration test that runs 18 queries against
  15 rows of deliberately awkward data (negatives, zero, boundary values) and
  asserts the **JIT-compiled machine code** and the **reference interpreter**
  agree on every single result. If Cranelift's codegen ever drifts from the
  interpreter's semantics, this is what catches it.

## Benchmarking

```bash
cargo bench --bench query_bench
```

Reports (via [Criterion](https://github.com/bheisler/criterion.rs)):
- `jit_filter_10k_rows` — time to run a compiled predicate over 10,000 rows
- `jit_compile_only` — one-time cost of parsing + Cranelift-compiling a query

On a typical laptop this comes out to roughly **~40µs to filter 10k rows**
and **~130µs to compile a query** — i.e. compiling is the expensive part,
filtering is fast, which is exactly the tradeoff a JIT is supposed to make
worthwhile when a query runs against many batches.

## Known limitations / roadmap

This is a demo/teaching-scale project, not a production query engine. Known
gaps, left as deliberate simplifications rather than bugs:

- **No unary minus.** `a < -10` doesn't parse — only `IDENT | INT | FLOAT |
  STRING | "(" expr ")"` are valid atoms. Column *values* can be negative
  (`a < 0` is fine); a negative *literal* in the query text isn't yet
  expressible.
- **Only `Int64`/`Float64` columns are JIT-compiled.** String/boolean/null
  columns are rejected with a clear error in `engine.rs` rather than
  mishandled — route those through `interp::eval_bool` instead.
- **Single-threaded, single-batch.** No parallelism across batches, no
  streaming, no query-plan caching beyond what one `Engine` already gives you
  (each compiled query gets a unique Cranelift symbol name so one `Engine`
  can safely compile many queries in a row).
- **CSV loader is intentionally minimal** — no quoted fields, no embedded
  commas, no null cells. Fine for clean numeric data; swap in `arrow-csv` if
  you need the full CSV spec.

## License

MIT — see `LICENSE`.
