

use std::fmt;

use crate::ast::{BinOp, Expr, UnOp};
use crate::lexer::{LexError, Lexer, Token};

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error: {}", self.message)
    }
}

impl std::error::Error for ParseError {}

impl From<LexError> for ParseError {
    fn from(e: LexError) -> Self {
        ParseError { message: e.to_string() }
    }
}

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

/// Parse a full query string into an `Expr`. Fails if there is trailing
/// input after a valid expression.
pub fn parse(input: &str) -> Result<Expr, ParseError> {
    let tokens = Lexer::new(input).tokenize()?;
    let mut parser = Parser { tokens, pos: 0 };
    let expr = parser.parse_expr()?;
    parser.expect(&Token::Eof)?;
    Ok(expr)
}

impl Parser {
    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn bump(&mut self) -> Token {
        let t = self.tokens[self.pos].clone();
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, want: &Token) -> Result<(), ParseError> {
        if self.peek() == want {
            self.bump();
            Ok(())
        } else {
            Err(ParseError { message: format!("expected {want}, found {}", self.peek()) })
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_and()?;
        while *self.peek() == Token::OrOr {
            self.bump();
            let rhs = self.parse_and()?;
            lhs = Expr::Binary(Box::new(lhs), BinOp::Or, Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_cmp()?;
        while *self.peek() == Token::AndAnd {
            self.bump();
            let rhs = self.parse_cmp()?;
            lhs = Expr::Binary(Box::new(lhs), BinOp::And, Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_cmp(&mut self) -> Result<Expr, ParseError> {
        let lhs = self.parse_unary()?;
        let op = match self.peek() {
            Token::Gt => BinOp::Gt,
            Token::Lt => BinOp::Lt,
            Token::Ge => BinOp::Ge,
            Token::Le => BinOp::Le,
            Token::EqEq => BinOp::Eq,
            Token::Ne => BinOp::Ne,
            _ => return Ok(lhs),
        };
        self.bump();
        let rhs = self.parse_unary()?;
        Ok(Expr::Binary(Box::new(lhs), op, Box::new(rhs)))
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        if *self.peek() == Token::Bang {
            self.bump();
            let inner = self.parse_unary()?;
            return Ok(Expr::Unary(UnOp::Not, Box::new(inner)));
        }
        self.parse_atom()
    }

    fn parse_atom(&mut self) -> Result<Expr, ParseError> {
        match self.bump() {
            Token::Ident(name) => Ok(Expr::Column(name)),
            Token::Int(n) => Ok(Expr::IntLit(n)),
            Token::Float(x) => Ok(Expr::FloatLit(x)),
            Token::Str(s) => Ok(Expr::StrLit(s)),
            Token::LParen => {
                let inner = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(inner)
            }
            other => Err(ParseError { message: format!("unexpected {other}, expected an expression") }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_comparison() {
        let e = parse("age > 10").unwrap();
        assert_eq!(
            e,
            Expr::Binary(Box::new(Expr::Column("age".into())), BinOp::Gt, Box::new(Expr::IntLit(10)))
        );
    }

    #[test]
    fn respects_and_over_or_precedence() {
        // a || b && c  ==  a || (b && c)
        let e = parse("a == 1 || b == 2 && c == 3").unwrap();
        let expected = Expr::Binary(
            Box::new(Expr::Binary(Box::new(Expr::Column("a".into())), BinOp::Eq, Box::new(Expr::IntLit(1)))),
            BinOp::Or,
            Box::new(Expr::Binary(
                Box::new(Expr::Binary(Box::new(Expr::Column("b".into())), BinOp::Eq, Box::new(Expr::IntLit(2)))),
                BinOp::And,
                Box::new(Expr::Binary(Box::new(Expr::Column("c".into())), BinOp::Eq, Box::new(Expr::IntLit(3)))),
            )),
        );
        assert_eq!(e, expected);
    }

    #[test]
    fn parses_parens_and_not() {
        let e = parse("!(a > 1 && b < 2)").unwrap();
        assert_eq!(
            e,
            Expr::Unary(
                UnOp::Not,
                Box::new(Expr::Binary(
                    Box::new(Expr::Binary(Box::new(Expr::Column("a".into())), BinOp::Gt, Box::new(Expr::IntLit(1)))),
                    BinOp::And,
                    Box::new(Expr::Binary(Box::new(Expr::Column("b".into())), BinOp::Lt, Box::new(Expr::IntLit(2)))),
                ))
            )
        );
    }

    #[test]
    fn rejects_trailing_garbage() {
        assert!(parse("a > 1 )").is_err());
    }

    #[test]
    fn rejects_dangling_and() {
        assert!(parse("a > 1 &&").is_err());
    }
}
