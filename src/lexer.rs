//! Hand-rolled tokenizer for the ExprJIT query language.
//!
//! Deliberately dependency-free (no `logos`) so this file can be compiled
//! and unit-tested in isolation without pulling in a crate whose exact
//! version/API might drift. Swap in `logos` later if you want derive-macro
//! ergonomics; the `Token` enum below is shaped to make that an easy
//! find-and-replace.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Ident(String),
    Int(i64),
    Float(f64),
    Str(String),

    Gt,
    Lt,
    Ge,
    Le,
    EqEq,
    Ne,
    AndAnd,
    OrOr,
    Bang,

    LParen,
    RParen,

    Eof,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Ident(s) => write!(f, "identifier `{s}`"),
            Token::Int(n) => write!(f, "integer `{n}`"),
            Token::Float(x) => write!(f, "float `{x}`"),
            Token::Str(s) => write!(f, "string {s:?}"),
            Token::Gt => write!(f, "`>`"),
            Token::Lt => write!(f, "`<`"),
            Token::Ge => write!(f, "`>=`"),
            Token::Le => write!(f, "`<=`"),
            Token::EqEq => write!(f, "`==`"),
            Token::Ne => write!(f, "`!=`"),
            Token::AndAnd => write!(f, "`&&`"),
            Token::OrOr => write!(f, "`||`"),
            Token::Bang => write!(f, "`!`"),
            Token::LParen => write!(f, "`(`"),
            Token::RParen => write!(f, "`)`"),
            Token::Eof => write!(f, "end of input"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LexError {
    pub message: String,
    pub pos: usize,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lex error at byte {}: {}", self.pos, self.message)
    }
}

impl std::error::Error for LexError {}

pub struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Self {
        Lexer { src: src.as_bytes(), pos: 0 }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<u8> {
        self.src.get(self.pos + 1).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_ascii_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn lex_ident(&mut self) -> Token {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == b'_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let s = std::str::from_utf8(&self.src[start..self.pos]).unwrap().to_string();
        Token::Ident(s)
    }

    fn lex_number(&mut self) -> Result<Token, LexError> {
        let start = self.pos;
        let mut is_float = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.pos += 1;
            } else if c == b'.' && !is_float && matches!(self.peek2(), Some(d) if d.is_ascii_digit()) {
                is_float = true;
                self.pos += 1;
            } else {
                break;
            }
        }
        let text = std::str::from_utf8(&self.src[start..self.pos]).unwrap();
        if is_float {
            text.parse::<f64>()
                .map(Token::Float)
                .map_err(|e| LexError { message: format!("invalid float literal: {e}"), pos: start })
        } else {
            text.parse::<i64>()
                .map(Token::Int)
                .map_err(|e| LexError { message: format!("invalid integer literal: {e}"), pos: start })
        }
    }

    fn lex_string(&mut self) -> Result<Token, LexError> {
        let quote_pos = self.pos;
        self.bump(); // consume opening quote
        let mut s = String::new();
        loop {
            match self.bump() {
                Some(b'"') => return Ok(Token::Str(s)),
                Some(b'\\') => match self.bump() {
                    Some(b'n') => s.push('\n'),
                    Some(b't') => s.push('\t'),
                    Some(b'"') => s.push('"'),
                    Some(b'\\') => s.push('\\'),
                    Some(c) => s.push(c as char),
                    None => {
                        return Err(LexError { message: "unterminated escape in string literal".into(), pos: quote_pos })
                    }
                },
                Some(c) => s.push(c as char),
                None => return Err(LexError { message: "unterminated string literal".into(), pos: quote_pos }),
            }
        }
    }

    pub fn next_token(&mut self) -> Result<Token, LexError> {
        self.skip_whitespace();
        let start = self.pos;
        let Some(c) = self.peek() else { return Ok(Token::Eof) };

        match c {
            b'(' => {
                self.bump();
                Ok(Token::LParen)
            }
            b')' => {
                self.bump();
                Ok(Token::RParen)
            }
            b'"' => self.lex_string(),
            b'0'..=b'9' => self.lex_number(),
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => Ok(self.lex_ident()),
            b'>' => {
                self.bump();
                if self.peek() == Some(b'=') {
                    self.bump();
                    Ok(Token::Ge)
                } else {
                    Ok(Token::Gt)
                }
            }
            b'<' => {
                self.bump();
                if self.peek() == Some(b'=') {
                    self.bump();
                    Ok(Token::Le)
                } else {
                    Ok(Token::Lt)
                }
            }
            b'=' => {
                self.bump();
                if self.peek() == Some(b'=') {
                    self.bump();
                    Ok(Token::EqEq)
                } else {
                    Err(LexError { message: "unexpected `=` (did you mean `==`?)".into(), pos: start })
                }
            }
            b'!' => {
                self.bump();
                if self.peek() == Some(b'=') {
                    self.bump();
                    Ok(Token::Ne)
                } else {
                    Ok(Token::Bang)
                }
            }
            b'&' => {
                self.bump();
                if self.peek() == Some(b'&') {
                    self.bump();
                    Ok(Token::AndAnd)
                } else {
                    Err(LexError { message: "unexpected `&` (did you mean `&&`?)".into(), pos: start })
                }
            }
            b'|' => {
                self.bump();
                if self.peek() == Some(b'|') {
                    self.bump();
                    Ok(Token::OrOr)
                } else {
                    Err(LexError { message: "unexpected `|` (did you mean `||`?)".into(), pos: start })
                }
            }
            other => Err(LexError { message: format!("unexpected character '{}'", other as char), pos: start }),
        }
    }

    /// Tokenize the entire input, ending with a single `Token::Eof`.
    pub fn tokenize(mut self) -> Result<Vec<Token>, LexError> {
        let mut out = Vec::new();
        loop {
            let tok = self.next_token()?;
            let is_eof = tok == Token::Eof;
            out.push(tok);
            if is_eof {
                break;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_basic_expr() {
        let toks = Lexer::new("age > 10 && score <= 5.5").tokenize().unwrap();
        assert_eq!(
            toks,
            vec![
                Token::Ident("age".into()),
                Token::Gt,
                Token::Int(10),
                Token::AndAnd,
                Token::Ident("score".into()),
                Token::Le,
                Token::Float(5.5),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn tokenizes_string_and_not() {
        let toks = Lexer::new(r#"!(name == "bob")"#).tokenize().unwrap();
        assert_eq!(
            toks,
            vec![
                Token::Bang,
                Token::LParen,
                Token::Ident("name".into()),
                Token::EqEq,
                Token::Str("bob".into()),
                Token::RParen,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn rejects_single_amp() {
        assert!(Lexer::new("a & b").tokenize().is_err());
    }
}
