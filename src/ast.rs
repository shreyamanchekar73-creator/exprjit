//! Abstract Syntax Tree for the ExprJIT query expression language.
//!
//! Supported grammar (informal):
//!   expr    := or_expr
//!   or_expr := and_expr ( "||" and_expr )*
//!   and_expr:= cmp_expr ( "&&" cmp_expr )*
//!   cmp_expr:= unary ( (">" | "<" | ">=" | "<=" | "==" | "!=") unary )?
//!   unary   := "!"* atom
//!   atom    := IDENT | INT | FLOAT | STRING | "(" expr ")"

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Reference to a column by name, e.g. `age`
    Column(String),
    IntLit(i64),
    FloatLit(f64),
    StrLit(String),
    Unary(UnOp, Box<Expr>),
    Binary(Box<Expr>, BinOp, Box<Expr>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    Gt,
    Lt,
    Ge,
    Le,
    Eq,
    Ne,
    And,
    Or,
}
