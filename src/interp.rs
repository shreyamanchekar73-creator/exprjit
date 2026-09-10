use std::collections::HashMap;
use std::fmt;

use crate::ast::{BinOp, Expr, UnOp};

/// A single column's value for one row, as seen by the interpreter.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvalError {
    pub message: String,
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "eval error: {}", self.message)
    }
}

impl std::error::Error for EvalError {}

fn err(msg: impl Into<String>) -> EvalError {
    EvalError { message: msg.into() }
}

/// One row's worth of column values, keyed by column name.
pub type Row<'a> = HashMap<&'a str, Value>;


pub fn eval_bool(expr: &Expr, row: &Row) -> Result<bool, EvalError> {
    match eval(expr, row)? {
        Value::Bool(b) => Ok(b),
        other => Err(err(format!("expected a boolean-valued expression, got {other:?}"))),
    }
}

fn eval(expr: &Expr, row: &Row) -> Result<Value, EvalError> {
    match expr {
        Expr::IntLit(n) => Ok(Value::Int(*n)),
        Expr::FloatLit(f) => Ok(Value::Float(*f)),
        Expr::StrLit(s) => Ok(Value::Str(s.clone())),
        Expr::Column(name) => row
            .get(name.as_str())
            .cloned()
            .ok_or_else(|| err(format!("unknown column referenced in query: {name}"))),
        Expr::Unary(UnOp::Not, inner) => {
            let b = eval_bool(inner, row)?;
            Ok(Value::Bool(!b))
        }
        Expr::Binary(lhs, op, rhs) => eval_binop(lhs, *op, rhs, row),
    }
}

fn eval_binop(lhs: &Expr, op: BinOp, rhs: &Expr, row: &Row) -> Result<Value, EvalError> {
    if matches!(op, BinOp::And | BinOp::Or) {
        let l = eval_bool(lhs, row)?;
        let r = eval_bool(rhs, row)?;
        return Ok(Value::Bool(match op {
            BinOp::And => l && r,
            BinOp::Or => l || r,
            _ => unreachable!(),
        }));
    }

    let lv = eval(lhs, row)?;
    let rv = eval(rhs, row)?;
    let ordering = compare(&lv, &rv)?;

    let result = match op {
        BinOp::Gt => ordering == std::cmp::Ordering::Greater,
        BinOp::Lt => ordering == std::cmp::Ordering::Less,
        BinOp::Ge => ordering != std::cmp::Ordering::Less,
        BinOp::Le => ordering != std::cmp::Ordering::Greater,
        BinOp::Eq => ordering == std::cmp::Ordering::Equal,
        BinOp::Ne => ordering != std::cmp::Ordering::Equal,
        BinOp::And | BinOp::Or => unreachable!("handled above"),
    };
    Ok(Value::Bool(result))
}

/// Compare two scalar values, promoting Int/Float to a common numeric type.
/// Strings compare lexicographically against strings only; comparing a
/// string against a number is a query error, not a silent `false`.
fn compare(lv: &Value, rv: &Value) -> Result<std::cmp::Ordering, EvalError> {
    use Value::*;
    match (lv, rv) {
        (Int(a), Int(b)) => Ok(a.cmp(b)),
        (Float(a), Float(b)) => a
            .partial_cmp(b)
            .ok_or_else(|| err("NaN comparison has no defined ordering")),
        (Int(a), Float(b)) => (*a as f64)
            .partial_cmp(b)
            .ok_or_else(|| err("NaN comparison has no defined ordering")),
        (Float(a), Int(b)) => a
            .partial_cmp(&(*b as f64))
            .ok_or_else(|| err("NaN comparison has no defined ordering")),
        (Str(a), Str(b)) => Ok(a.cmp(b)),
        (a, b) => Err(err(format!("cannot compare {a:?} with {b:?}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn row_of(pairs: &[(&'static str, Value)]) -> Row<'static> {
        pairs.iter().cloned().collect()
    }

    #[test]
    fn evaluates_numeric_comparison() {
        let expr = parse("age > 10").unwrap();
        let row = row_of(&[("age", Value::Int(15))]);
        assert!(eval_bool(&expr, &row).unwrap());

        let row = row_of(&[("age", Value::Int(5))]);
        assert!(!eval_bool(&expr, &row).unwrap());
    }

    #[test]
    fn evaluates_and_or_not() {
        let expr = parse("!(a > 1 && b < 2)").unwrap();
        let row = row_of(&[("a", Value::Int(5)), ("b", Value::Int(1))]);
        // a > 1 (true) && b < 2 (true) = true; negated = false
        assert!(!eval_bool(&expr, &row).unwrap());

        let row = row_of(&[("a", Value::Int(0)), ("b", Value::Int(1))]);
        // a > 1 (false) && ... = false; negated = true
        assert!(eval_bool(&expr, &row).unwrap());
    }

    #[test]
    fn mixed_int_float_comparison() {
        let expr = parse("score >= 4.5").unwrap();
        let row = row_of(&[("score", Value::Int(5))]);
        assert!(eval_bool(&expr, &row).unwrap());
    }

    #[test]
    fn string_equality() {
        let expr = parse(r#"name == "bob""#).unwrap();
        let row = row_of(&[("name", Value::Str("bob".into()))]);
        assert!(eval_bool(&expr, &row).unwrap());
    }

    #[test]
    fn unknown_column_errors() {
        let expr = parse("nope > 1").unwrap();
        let row = row_of(&[]);
        assert!(eval_bool(&expr, &row).is_err());
    }
}
