use std::collections::HashMap;

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::ast::{BinOp, Expr, UnOp};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColType {
    Int64,
    Float64,
}

/// Maps a column name to its (index-in-batch, type) so the codegen knows how
/// to load it.
#[derive(Debug, Clone, Default)]
pub struct Schema {
    pub columns: HashMap<String, (usize, ColType)>,
}

/// Signature of a compiled per-row predicate.
pub type CompiledFilterFn = unsafe extern "C" fn(*const *const u8, i64) -> u8;

/// Signature of a compiled *whole-loop* filter:
/// `fn(columns, n_rows, out) -> match_count`.
/// `out` must have room for `n_rows` u32 values; the first `match_count`
/// entries are the matching row indexes.
pub type CompiledLoopFn = unsafe extern "C" fn(*const *const u8, i64, *mut u32) -> i64;

pub struct Jit {
    module: JITModule,
}
impl Default for Jit {
    fn default() -> Self {
        Self::new()
    }
}

impl Jit {
    pub fn new() -> Self {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();
        // Speed over safety checks for the JIT'd predicate itself; the outer
        // Rust code is still fully safe/bounds-checked.
        flag_builder.set("opt_level", "speed").unwrap();

        let isa_builder =
            cranelift_native::builder().expect("host machine not supported by Cranelift");
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .expect("failed to build target ISA");

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);
        Jit { module }
    }

    /// Compile `expr` against `schema` into a callable native function.
    pub fn compile(&mut self, expr: &Expr, schema: &Schema, fn_name: &str) -> CompiledFilterFn {
        let mut ctx = self.module.make_context();
        let mut fn_builder_ctx = FunctionBuilderContext::new();

        let ptr_type = self.module.target_config().pointer_type();
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // columns: *const *const u8
        ctx.func.signature.params.push(AbiParam::new(types::I64)); // row_idx: i64
        ctx.func.signature.returns.push(AbiParam::new(types::I8)); // -> bool (0/1)

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let cols_ptr = builder.block_params(entry_block)[0];
            let row_idx = builder.block_params(entry_block)[1];

            let mut codegen = ExprCodegen {
                builder: &mut builder,
                schema,
                cols_ptr,
                row_idx,
                ptr_type,
            };
            let result = codegen.gen_bool_value(expr);
            builder.ins().return_(&[result]);
            builder.finalize();
        }

        let func_id = self
            .module
            .declare_function(fn_name, Linkage::Export, &ctx.func.signature)
            .expect("failed to declare function");
        self.module
            .define_function(func_id, &mut ctx)
            .expect("failed to define function");
        self.module.clear_context(&mut ctx);
        self.module
            .finalize_definitions()
            .expect("failed to finalize JIT definitions");

        let code_ptr = self.module.get_finalized_function(func_id);
        unsafe { std::mem::transmute::<*const u8, CompiledFilterFn>(code_ptr) }
    }

    /// Compile `expr` into a native function that runs the *entire* filter
    /// loop: for every row it evaluates the predicate and records the row
    /// index when it matches. No Rust code runs between rows.
    pub fn compile_loop(&mut self, expr: &Expr, schema: &Schema, fn_name: &str) -> CompiledLoopFn {
        let mut ctx = self.module.make_context();
        let mut fn_builder_ctx = FunctionBuilderContext::new();

        let ptr_type = self.module.target_config().pointer_type();
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // columns
        ctx.func.signature.params.push(AbiParam::new(types::I64)); // n_rows
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // out buffer
        ctx.func.signature.returns.push(AbiParam::new(types::I64)); // match count

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
            let entry = builder.create_block();
            let header = builder.create_block();
            let body = builder.create_block();
            let exit = builder.create_block();

            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);

            let cols_ptr = builder.block_params(entry)[0];
            let n_rows = builder.block_params(entry)[1];
            let out_ptr = builder.block_params(entry)[2];

            // Loop state lives in Cranelift variables (SSA is built for us).
            let i_var = builder.declare_var(types::I64);
            let count_var = builder.declare_var(types::I64);
            let zero = builder.ins().iconst(types::I64, 0);
            builder.def_var(i_var, zero);
            builder.def_var(count_var, zero);
            builder.ins().jump(header, &[]);

            // header: while i < n_rows
            builder.switch_to_block(header);
            let i = builder.use_var(i_var);
            let more = builder.ins().icmp(IntCC::SignedLessThan, i, n_rows);
            builder.ins().brif(more, body, &[], exit, &[]);

            // body: evaluate predicate for row i, record it if it matches
            builder.switch_to_block(body);
            let i = builder.use_var(i_var);
            let count = builder.use_var(count_var);
            let keep = {
                let mut codegen = ExprCodegen {
                    builder: &mut builder,
                    schema,
                    cols_ptr,
                    row_idx: i,
                    ptr_type,
                };
                codegen.gen_bool_value(expr)
            };
            // Branch-free append: always write out[count] = i, then
            // count += keep. (count <= i < n_rows, so the write is in range.)
            let idx32 = builder.ins().ireduce(types::I32, i);
            let byte_off = builder.ins().imul_imm(count, 4);
            let addr = builder.ins().iadd(out_ptr, byte_off);
            builder.ins().store(MemFlags::trusted(), idx32, addr, 0);
            let keep64 = builder.ins().uextend(types::I64, keep);
            let new_count = builder.ins().iadd(count, keep64);
            let next_i = builder.ins().iadd_imm(i, 1);
            builder.def_var(count_var, new_count);
            builder.def_var(i_var, next_i);
            builder.ins().jump(header, &[]);

            // exit: return the number of matches
            builder.switch_to_block(exit);
            let result = builder.use_var(count_var);
            builder.ins().return_(&[result]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        let func_id = self
            .module
            .declare_function(fn_name, Linkage::Export, &ctx.func.signature)
            .expect("failed to declare function");
        self.module
            .define_function(func_id, &mut ctx)
            .expect("failed to define function");
        self.module.clear_context(&mut ctx);
        self.module
            .finalize_definitions()
            .expect("failed to finalize JIT definitions");

        let code_ptr = self.module.get_finalized_function(func_id);
        unsafe { std::mem::transmute::<*const u8, CompiledLoopFn>(code_ptr) }
    }
}

/// Intermediate typed value produced while walking the AST.
#[derive(Clone, Copy)]
enum Val {
    Bool(Value),
    Int(Value),
    Float(Value),
}

struct ExprCodegen<'a, 'b> {
    builder: &'a mut FunctionBuilder<'b>,
    schema: &'a Schema,
    cols_ptr: Value,
    row_idx: Value,
    ptr_type: Type,
}

impl<'a, 'b> ExprCodegen<'a, 'b> {
    fn gen(&mut self, expr: &Expr) -> Val {
        match expr {
            Expr::IntLit(n) => Val::Int(self.builder.ins().iconst(types::I64, *n)),
            Expr::FloatLit(f) => Val::Float(self.builder.ins().f64const(*f)),
            Expr::StrLit(_) => {
                panic!("string comparisons are not lowered by the numeric JIT path in this demo")
            }
            Expr::Column(name) => self.gen_column_load(name),
            Expr::Unary(UnOp::Not, inner) => {
                let v = self.gen_bool_value(inner);
                let one = self.builder.ins().iconst(types::I8, 1);
                Val::Bool(self.builder.ins().bxor(v, one))
            }
            Expr::Binary(lhs, op, rhs) => self.gen_binop(lhs, *op, rhs),
        }
    }

    /// Evaluate `expr` and require the result to be boolean, returning the
    /// raw Cranelift `Value` (i8, 0 or 1).
    fn gen_bool_value(&mut self, expr: &Expr) -> Value {
        match self.gen(expr) {
            Val::Bool(v) => v,
            _ => panic!("expected a boolean-valued (comparison/logical) expression"),
        }
    }

    fn gen_column_load(&mut self, name: &str) -> Val {
        let (idx, ty) = *self
            .schema
            .columns
            .get(name)
            .unwrap_or_else(|| panic!("unknown column referenced in query: {name}"));

        // columns[idx] -> pointer to that column's contiguous buffer
        let ptr_bytes = self.ptr_type.bytes() as i32;
        let offset = (idx as i32) * ptr_bytes;
        let col_ptr =
            self.builder
                .ins()
                .load(self.ptr_type, MemFlags::trusted(), self.cols_ptr, offset);

        // address = col_ptr + row_idx * sizeof(element)
        let byte_offset = self.builder.ins().imul_imm(self.row_idx, 8); // both Int64/Float64 are 8 bytes
        let addr = self.builder.ins().iadd(col_ptr, byte_offset);

        match ty {
            ColType::Int64 => Val::Int(self.builder.ins().load(
                types::I64,
                MemFlags::trusted(),
                addr,
                0,
            )),
            ColType::Float64 => Val::Float(self.builder.ins().load(
                types::F64,
                MemFlags::trusted(),
                addr,
                0,
            )),
        }
    }

    fn coerce_float(&mut self, v: Val) -> Value {
        match v {
            Val::Float(f) => f,
            Val::Int(i) => self.builder.ins().fcvt_from_sint(types::F64, i),
            Val::Bool(_) => panic!("cannot use a boolean value in a numeric comparison"),
        }
    }

    fn gen_binop(&mut self, lhs: &Expr, op: BinOp, rhs: &Expr) -> Val {
        match op {
            BinOp::And | BinOp::Or => {
                let l = self.gen_bool_value(lhs);
                let r = self.gen_bool_value(rhs);
                let v = match op {
                    BinOp::And => self.builder.ins().band(l, r),
                    BinOp::Or => self.builder.ins().bor(l, r),
                    _ => unreachable!(),
                };
                Val::Bool(v)
            }
            _ => {
                let lv = self.gen(lhs);
                let rv = self.gen(rhs);
                let both_int = matches!((&lv, &rv), (Val::Int(_), Val::Int(_)));

                let cmp_result = if both_int {
                    let (Val::Int(l), Val::Int(r)) = (lv, rv) else {
                        unreachable!()
                    };
                    let cc = match op {
                        BinOp::Gt => IntCC::SignedGreaterThan,
                        BinOp::Lt => IntCC::SignedLessThan,
                        BinOp::Ge => IntCC::SignedGreaterThanOrEqual,
                        BinOp::Le => IntCC::SignedLessThanOrEqual,
                        BinOp::Eq => IntCC::Equal,
                        BinOp::Ne => IntCC::NotEqual,
                        _ => unreachable!(),
                    };
                    self.builder.ins().icmp(cc, l, r)
                } else {
                    let l = self.coerce_float(lv);
                    let r = self.coerce_float(rv);
                    let cc = match op {
                        BinOp::Gt => FloatCC::GreaterThan,
                        BinOp::Lt => FloatCC::LessThan,
                        BinOp::Ge => FloatCC::GreaterThanOrEqual,
                        BinOp::Le => FloatCC::LessThanOrEqual,
                        BinOp::Eq => FloatCC::Equal,
                        BinOp::Ne => FloatCC::NotEqual,
                        _ => unreachable!(),
                    };
                    self.builder.ins().fcmp(cc, l, r)
                };
                // icmp/fcmp already yield an i8 0/1 value on modern Cranelift.
                Val::Bool(cmp_result)
            }
        }
    }
}
