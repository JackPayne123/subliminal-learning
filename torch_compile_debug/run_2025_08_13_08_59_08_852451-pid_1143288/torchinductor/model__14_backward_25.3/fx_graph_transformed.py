class GraphModule(torch.nn.Module):
    def forward(self, primals_7: "Sym(s5)", primals_8: "Sym(s6)", primals_2: "Sym(17)", primals_3: "Sym(s1)", floordiv: "Sym(((s5*s6 + 8)//9))", sym_size_int_9: "Sym(s5*s6 - 8*(((s5*s6 + 8)//9)))", mm: "bf16[((s5*s6 + 8)//9), 262208]", amax: "f32[((s5*s6 + 8)//9), 1]", log: "f32[((s5*s6 + 8)//9), 1]", mm_1: "bf16[((s5*s6 + 8)//9), 262208]", amax_1: "f32[((s5*s6 + 8)//9), 1]", log_1: "f32[((s5*s6 + 8)//9), 1]", mm_2: "bf16[((s5*s6 + 8)//9), 262208]", amax_2: "f32[((s5*s6 + 8)//9), 1]", log_2: "f32[((s5*s6 + 8)//9), 1]", mm_3: "bf16[((s5*s6 + 8)//9), 262208]", amax_3: "f32[((s5*s6 + 8)//9), 1]", log_3: "f32[((s5*s6 + 8)//9), 1]", mm_4: "bf16[((s5*s6 + 8)//9), 262208]", amax_4: "f32[((s5*s6 + 8)//9), 1]", log_4: "f32[((s5*s6 + 8)//9), 1]", mm_5: "bf16[((s5*s6 + 8)//9), 262208]", amax_5: "f32[((s5*s6 + 8)//9), 1]", log_5: "f32[((s5*s6 + 8)//9), 1]", mm_6: "bf16[((s5*s6 + 8)//9), 262208]", amax_6: "f32[((s5*s6 + 8)//9), 1]", log_6: "f32[((s5*s6 + 8)//9), 1]", mm_7: "bf16[((s5*s6 + 8)//9), 262208]", amax_7: "f32[((s5*s6 + 8)//9), 1]", log_7: "f32[((s5*s6 + 8)//9), 1]", mm_8: "bf16[s5*s6 - 8*(((s5*s6 + 8)//9)), 262208]", amax_8: "f32[s5*s6 - 8*(((s5*s6 + 8)//9)), 1]", log_8: "f32[s5*s6 - 8*(((s5*s6 + 8)//9)), 1]", convert_element_type_36: "f32[]", ne_57: "b8[17*s1 - 8*(((17*s1 + 8)//9)), 1]", where_18: "i64[17*s1 - 8*(((17*s1 + 8)//9)), 1]", permute_10: "bf16[262208, 2560]", ne_59: "b8[((17*s1 + 8)//9), 1]", where_20: "i64[((17*s1 + 8)//9), 1]", ne_61: "b8[((17*s1 + 8)//9), 1]", where_22: "i64[((17*s1 + 8)//9), 1]", ne_63: "b8[((17*s1 + 8)//9), 1]", where_24: "i64[((17*s1 + 8)//9), 1]", ne_65: "b8[((17*s1 + 8)//9), 1]", where_26: "i64[((17*s1 + 8)//9), 1]", ne_67: "b8[((17*s1 + 8)//9), 1]", where_28: "i64[((17*s1 + 8)//9), 1]", ne_69: "b8[((17*s1 + 8)//9), 1]", where_30: "i64[((17*s1 + 8)//9), 1]", ne_71: "b8[((17*s1 + 8)//9), 1]", where_32: "i64[((17*s1 + 8)//9), 1]", ne_73: "b8[((17*s1 + 8)//9), 1]", where_34: "i64[((17*s1 + 8)//9), 1]", tangents_1: "f32[]"):
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:560 in _unsloth_compiled_fused_ce_loss_function, code: loss = loss / torch.tensor(divisor, dtype = torch.float32)
        div_1: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_36);  tangents_1 = convert_element_type_36 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        scatter_upon_const_tensor: "f32[s5*s6 - 8*(((s5*s6 + 8)//9)), 262208]" = torch__inductor_fx_passes_post_grad_scatter_upon_const_tensor(shape = [sym_size_int_9, 262208], background_val = 0, dtype = torch.float32, dim = 1, selector = where_18, val = -1.0);  sym_size_int_9 = where_18 = None
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19: "f32[17*s1 - 8*(((17*s1 + 8)//9)), 1]" = torch.ops.aten.where.self(ne_57, div_1, full_default_3);  ne_57 = None
        mul_149: "f32[s5*s6 - 8*(((s5*s6 + 8)//9)), 262208]" = torch.ops.aten.mul.Tensor(scatter_upon_const_tensor, where_19);  scatter_upon_const_tensor = where_19 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_34: "f32[s5*s6 - 8*(((s5*s6 + 8)//9)), 262208]" = torch.ops.prims.convert_element_type.default(mm_8, torch.float32);  mm_8 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_96: "f32[s5*s6 - 8*(((s5*s6 + 8)//9)), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_34, amax_8);  convert_element_type_34 = amax_8 = None
        sub_97: "f32[s5*s6 - 8*(((s5*s6 + 8)//9)), 262208]" = torch.ops.aten.sub.Tensor(sub_96, log_8);  sub_96 = log_8 = None
        exp_9: "f32[s5*s6 - 8*(((s5*s6 + 8)//9)), 262208]" = torch.ops.aten.exp.default(sub_97);  sub_97 = None
        sum_28: "f32[s5*s6 - 8*(((s5*s6 + 8)//9)), 1]" = torch.ops.aten.sum.dim_IntList(mul_149, [1], True)
        mul_150: "f32[s5*s6 - 8*(((s5*s6 + 8)//9)), 262208]" = torch.ops.aten.mul.Tensor(exp_9, sum_28);  exp_9 = sum_28 = None
        sub_100: "f32[s5*s6 - 8*(((s5*s6 + 8)//9)), 262208]" = torch.ops.aten.sub.Tensor(mul_149, mul_150);  mul_149 = mul_150 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_37: "bf16[s5*s6 - 8*(((s5*s6 + 8)//9)), 262208]" = torch.ops.prims.convert_element_type.default(sub_100, torch.bfloat16);  sub_100 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_9: "bf16[s5*s6 - 8*(((s5*s6 + 8)//9)), 2560]" = torch.ops.aten.mm.default(convert_element_type_37, permute_10);  convert_element_type_37 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        full_1: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.full.default([floordiv, 262208], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  floordiv = None
        scatter_1: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.scatter.value(full_1, 1, where_20, -1.0);  where_20 = None
        where_21: "f32[((17*s1 + 8)//9), 1]" = torch.ops.aten.where.self(ne_59, div_1, full_default_3);  ne_59 = None
        mul_151: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mul.Tensor(scatter_1, where_21);  scatter_1 = where_21 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_30: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_90: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_30, amax_7);  convert_element_type_30 = amax_7 = None
        sub_91: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_90, log_7);  sub_90 = log_7 = None
        exp_10: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_91);  sub_91 = None
        sum_29: "f32[((s5*s6 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(mul_151, [1], True)
        mul_152: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mul.Tensor(exp_10, sum_29);  exp_10 = sum_29 = None
        sub_101: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_40: "bf16[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(sub_101, torch.bfloat16);  sub_101 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_10: "bf16[((s5*s6 + 8)//9), 2560]" = torch.ops.aten.mm.default(convert_element_type_40, permute_10);  convert_element_type_40 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        scatter_2: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.scatter.value(full_1, 1, where_22, -1.0);  where_22 = None
        where_23: "f32[((17*s1 + 8)//9), 1]" = torch.ops.aten.where.self(ne_61, div_1, full_default_3);  ne_61 = None
        mul_153: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mul.Tensor(scatter_2, where_23);  scatter_2 = where_23 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_26: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_84: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_26, amax_6);  convert_element_type_26 = amax_6 = None
        sub_85: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_84, log_6);  sub_84 = log_6 = None
        exp_11: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_85);  sub_85 = None
        sum_30: "f32[((s5*s6 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(mul_153, [1], True)
        mul_154: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mul.Tensor(exp_11, sum_30);  exp_11 = sum_30 = None
        sub_102: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(mul_153, mul_154);  mul_153 = mul_154 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_43: "bf16[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(sub_102, torch.bfloat16);  sub_102 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_11: "bf16[((s5*s6 + 8)//9), 2560]" = torch.ops.aten.mm.default(convert_element_type_43, permute_10);  convert_element_type_43 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        scatter_3: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.scatter.value(full_1, 1, where_24, -1.0);  where_24 = None
        where_25: "f32[((17*s1 + 8)//9), 1]" = torch.ops.aten.where.self(ne_63, div_1, full_default_3);  ne_63 = None
        mul_155: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mul.Tensor(scatter_3, where_25);  scatter_3 = where_25 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_22: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_78: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_22, amax_5);  convert_element_type_22 = amax_5 = None
        sub_79: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_78, log_5);  sub_78 = log_5 = None
        exp_12: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_79);  sub_79 = None
        sum_31: "f32[((s5*s6 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(mul_155, [1], True)
        mul_156: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mul.Tensor(exp_12, sum_31);  exp_12 = sum_31 = None
        sub_103: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_46: "bf16[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(sub_103, torch.bfloat16);  sub_103 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_12: "bf16[((s5*s6 + 8)//9), 2560]" = torch.ops.aten.mm.default(convert_element_type_46, permute_10);  convert_element_type_46 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        scatter_4: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.scatter.value(full_1, 1, where_26, -1.0);  where_26 = None
        where_27: "f32[((17*s1 + 8)//9), 1]" = torch.ops.aten.where.self(ne_65, div_1, full_default_3);  ne_65 = None
        mul_157: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mul.Tensor(scatter_4, where_27);  scatter_4 = where_27 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_18: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_72: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_18, amax_4);  convert_element_type_18 = amax_4 = None
        sub_73: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_72, log_4);  sub_72 = log_4 = None
        exp_13: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_73);  sub_73 = None
        sum_32: "f32[((s5*s6 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(mul_157, [1], True)
        mul_158: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mul.Tensor(exp_13, sum_32);  exp_13 = sum_32 = None
        sub_104: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(mul_157, mul_158);  mul_157 = mul_158 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_49: "bf16[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(sub_104, torch.bfloat16);  sub_104 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_13: "bf16[((s5*s6 + 8)//9), 2560]" = torch.ops.aten.mm.default(convert_element_type_49, permute_10);  convert_element_type_49 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        scatter_5: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.scatter.value(full_1, 1, where_28, -1.0);  where_28 = None
        where_29: "f32[((17*s1 + 8)//9), 1]" = torch.ops.aten.where.self(ne_67, div_1, full_default_3);  ne_67 = None
        mul_159: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mul.Tensor(scatter_5, where_29);  scatter_5 = where_29 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_14: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_66: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_14, amax_3);  convert_element_type_14 = amax_3 = None
        sub_67: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_66, log_3);  sub_66 = log_3 = None
        exp_14: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
        sum_33: "f32[((s5*s6 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(mul_159, [1], True)
        mul_160: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mul.Tensor(exp_14, sum_33);  exp_14 = sum_33 = None
        sub_105: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(mul_159, mul_160);  mul_159 = mul_160 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_52: "bf16[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(sub_105, torch.bfloat16);  sub_105 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_14: "bf16[((s5*s6 + 8)//9), 2560]" = torch.ops.aten.mm.default(convert_element_type_52, permute_10);  convert_element_type_52 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        scatter_6: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.scatter.value(full_1, 1, where_30, -1.0);  where_30 = None
        where_31: "f32[((17*s1 + 8)//9), 1]" = torch.ops.aten.where.self(ne_69, div_1, full_default_3);  ne_69 = None
        mul_161: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mul.Tensor(scatter_6, where_31);  scatter_6 = where_31 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_10: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_60: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_10, amax_2);  convert_element_type_10 = amax_2 = None
        sub_61: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_60, log_2);  sub_60 = log_2 = None
        exp_15: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        sum_34: "f32[((s5*s6 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(mul_161, [1], True)
        mul_162: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mul.Tensor(exp_15, sum_34);  exp_15 = sum_34 = None
        sub_106: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(mul_161, mul_162);  mul_161 = mul_162 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_55: "bf16[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(sub_106, torch.bfloat16);  sub_106 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_15: "bf16[((s5*s6 + 8)//9), 2560]" = torch.ops.aten.mm.default(convert_element_type_55, permute_10);  convert_element_type_55 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        scatter_7: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.scatter.value(full_1, 1, where_32, -1.0);  where_32 = None
        where_33: "f32[((17*s1 + 8)//9), 1]" = torch.ops.aten.where.self(ne_71, div_1, full_default_3);  ne_71 = None
        mul_163: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mul.Tensor(scatter_7, where_33);  scatter_7 = where_33 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_6: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_54: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_6, amax_1);  convert_element_type_6 = amax_1 = None
        sub_55: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_54, log_1);  sub_54 = log_1 = None
        exp_16: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_35: "f32[((s5*s6 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(mul_163, [1], True)
        mul_164: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mul.Tensor(exp_16, sum_35);  exp_16 = sum_35 = None
        sub_107: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(mul_163, mul_164);  mul_163 = mul_164 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_58: "bf16[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(sub_107, torch.bfloat16);  sub_107 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_16: "bf16[((s5*s6 + 8)//9), 2560]" = torch.ops.aten.mm.default(convert_element_type_58, permute_10);  convert_element_type_58 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        scatter_8: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.scatter.value(full_1, 1, where_34, -1.0);  full_1 = where_34 = None
        where_35: "f32[((17*s1 + 8)//9), 1]" = torch.ops.aten.where.self(ne_73, div_1, full_default_3);  ne_73 = div_1 = full_default_3 = None
        mul_165: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mul.Tensor(scatter_8, where_35);  scatter_8 = where_35 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_2: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_48: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        sub_49: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_48, log);  sub_48 = log = None
        exp_17: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_36: "f32[((s5*s6 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(mul_165, [1], True)
        mul_166: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mul.Tensor(exp_17, sum_36);  exp_17 = sum_36 = None
        sub_108: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_61: "bf16[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(sub_108, torch.bfloat16);  sub_108 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_17: "bf16[((s5*s6 + 8)//9), 2560]" = torch.ops.aten.mm.default(convert_element_type_61, permute_10);  convert_element_type_61 = permute_10 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:530 in _unsloth_compiled_fused_ce_loss_function, code: __shift_states = torch.chunk(hidden_states, n_chunks, dim = 0)
        cat: "bf16[s5*s6, 2560]" = torch.ops.aten.cat.default([mm_17, mm_16, mm_15, mm_14, mm_13, mm_12, mm_11, mm_10, mm_9]);  mm_17 = mm_16 = mm_15 = mm_14 = mm_13 = mm_12 = mm_11 = mm_10 = mm_9 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:528 in _unsloth_compiled_fused_ce_loss_function, code: hidden_states = hidden_states.view(-1, hd)
        view_11: "bf16[s5, s6, 2560]" = torch.ops.aten.reshape.default(cat, [primals_7, primals_8, 2560]);  cat = primals_7 = primals_8 = None
        return (None, None, None, None, None, None, None, None, view_11, None)
        