window.EXPRJIT_DATA = {
    query: "age>18",
    totalRows: 20,
    matchedRows: 1,
    matchPct: 5.00,
    coldCompileUs: 1385.50,
    warmCompileUs: 0,
    compileUs: 0.00,
    jitUs: 7.00,
    interpreterUs: 4.00,
    speedup: 0.57,
    cacheStatus: "HIT",
    correctness: "PASS",
    compiledShapes: 1,
    indices: [19],
    history: [
        { query: "age>18", rows: 20, matched: 1, cache: "MISS", jitUs: 13.90, interpreterUs: 6.50 },
        { query: "age>18", rows: 20, matched: 1, cache: "HIT", jitUs: 7.00, interpreterUs: 4.00 }
    ]
};
