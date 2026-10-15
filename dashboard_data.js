window.EXPRJIT_DATA = {
    query: "age > 10 && score < 20.0",
    totalRows: 20,
    matchedRows: 5,
    matchPct: 25.00,
    coldCompileUs: 922.00,
    warmCompileUs: 0,
    compileUs: 0.00,
    jitUs: 6.60,
    interpreterUs: 7.20,
    speedup: 1.09,
    cacheStatus: "HIT",
    correctness: "PASS",
    compiledShapes: 1,
    indices: [11, 12, 13, 14, 15],
    history: [
        { query: "age > 10 && score < 20.0", rows: 20, matched: 5, cache: "MISS", jitUs: 12.50, interpreterUs: 7.40 },
        { query: "age > 10 && score < 20.0", rows: 20, matched: 5, cache: "HIT", jitUs: 6.60, interpreterUs: 7.20 }
    ]
};
