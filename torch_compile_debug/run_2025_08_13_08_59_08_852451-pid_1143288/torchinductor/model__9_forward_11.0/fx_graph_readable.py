class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[262208, 2560]", primals_2: "Sym(s0)", primals_3: "i64[22, s0]", primals_4: "Sym(s1)", primals_5: "i64[22, s1]", primals_6: "Sym(s4)", primals_7: "bf16[22, s4, 2560]", primals_8: "i64[]"):
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:517 in _unsloth_compiled_fused_ce_loss_function, code: shift_labels = torch.empty_like(output_labels, device = device)
        empty: "i64[22, s0]" = torch.ops.aten.empty.memory_format([22, primals_2], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute: "i64[22, s0]" = torch.ops.aten.permute.default(empty, [0, 1]);  empty = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:518 in _unsloth_compiled_fused_ce_loss_function, code: shift_labels[..., :-1] = output_labels[..., 1:]
        slice_1: "i64[22, s0 - 1]" = torch.ops.aten.slice.Tensor(primals_3, 1, 1, 9223372036854775807);  primals_3 = None
        slice_scatter: "i64[22, s0]" = torch.ops.aten.slice_scatter.default(permute, slice_1, 1, 0, -1);  slice_1 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:521 in _unsloth_compiled_fused_ce_loss_function, code: shift_labels[..., :-1][mask[..., 1:] == 0] = -100
        slice_5: "i64[22, s1 - 1]" = torch.ops.aten.slice.Tensor(primals_5, 1, 1, 9223372036854775807);  primals_5 = None
        eq_16: "b8[22, s1 - 1]" = torch.ops.aten.eq.Scalar(slice_5, 0);  slice_5 = None
        full_default: "i64[]" = torch.ops.aten.full.default([], -100, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        slice_6: "i64[22, s0 - 1]" = torch.ops.aten.slice.Tensor(slice_scatter, 1, 0, -1)
        index_put: "i64[22, s0 - 1]" = torch.ops.aten.index_put.default(slice_6, [eq_16], full_default);  slice_6 = eq_16 = None
        slice_scatter_1: "i64[22, s0]" = torch.ops.aten.slice_scatter.default(slice_scatter, index_put, 1, 0, -1);  slice_scatter = index_put = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:523 in _unsloth_compiled_fused_ce_loss_function, code: shift_labels[..., -1] = -100
        select_1: "i64[22]" = torch.ops.aten.select.int(slice_scatter_1, 1, -1)
        copy_1: "i64[22]" = torch.ops.aten.copy.default(select_1, full_default);  select_1 = full_default = None
        select_scatter: "i64[22, s0]" = torch.ops.aten.select_scatter.default(slice_scatter_1, copy_1, 1, -1);  slice_scatter_1 = copy_1 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:524 in _unsloth_compiled_fused_ce_loss_function, code: shift_labels = shift_labels.view(-1)
        sym_numel_default: "Sym(22*s0)" = torch.ops.aten.sym_numel.default(permute);  permute = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:528 in _unsloth_compiled_fused_ce_loss_function, code: hidden_states = hidden_states.view(-1, hd)
        view_1: "bf16[22*s4, 2560]" = torch.ops.aten.view.default(primals_7, [-1, 2560]);  primals_7 = None
        sym_size_int_7: "Sym(22*s4)" = torch.ops.aten.sym_size.int(view_1, 0)
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:530 in _unsloth_compiled_fused_ce_loss_function, code: __shift_states = torch.chunk(hidden_states, n_chunks, dim = 0)
        add_47: "Sym(22*s4 + 9)" = sym_size_int_7 + 9;  sym_size_int_7 = None
        sub_14: "Sym(22*s4 + 8)" = add_47 - 1;  add_47 = None
        floordiv: "Sym(((22*s4 + 8)//9))" = sub_14 // 9;  sub_14 = None
        split = torch.ops.aten.split.Tensor(view_1, floordiv);  view_1 = None
        getitem: "bf16[((22*s4 + 8)//9), 2560]" = split[0]
        getitem_1: "bf16[((22*s4 + 8)//9), 2560]" = split[1]
        getitem_2: "bf16[((22*s4 + 8)//9), 2560]" = split[2]
        getitem_3: "bf16[((22*s4 + 8)//9), 2560]" = split[3]
        getitem_4: "bf16[((22*s4 + 8)//9), 2560]" = split[4]
        getitem_5: "bf16[((22*s4 + 8)//9), 2560]" = split[5]
        getitem_6: "bf16[((22*s4 + 8)//9), 2560]" = split[6]
        getitem_7: "bf16[((22*s4 + 8)//9), 2560]" = split[7]
        getitem_8: "bf16[22*s4 - 8*(((22*s4 + 8)//9)), 2560]" = split[8];  split = None
        sym_size_int_8: "Sym(22*s4 - 8*(((22*s4 + 8)//9)))" = torch.ops.aten.sym_size.int(getitem_8, 0)
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:531 in _unsloth_compiled_fused_ce_loss_function, code: __shift_labels = torch.chunk(shift_labels,  n_chunks, dim = 0)
        add_75: "Sym(22*s0 + 9)" = sym_numel_default + 9;  sym_numel_default = None
        sub_24: "Sym(22*s0 + 8)" = add_75 - 1;  add_75 = None
        floordiv_1: "Sym(((22*s0 + 8)//9))" = sub_24 // 9;  sub_24 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        permute_1: "bf16[2560, 262208]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        mm: "bf16[((22*s4 + 8)//9), 262208]" = torch.ops.aten.mm.default(getitem, permute_1);  getitem = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_2: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm, torch.float32)
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        amax: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.amax.default(convert_element_type_2, [1], True)
        sub_36: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = None
        exp: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_36)
        sum_1: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_37: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_36, log);  sub_36 = None
        view_2: "i64[22*s0]" = torch.ops.aten.view.default(select_scatter, [-1]);  select_scatter = None
        split_2 = torch.ops.aten.split.Tensor(view_2, floordiv_1);  view_2 = floordiv_1 = None
        getitem_18: "i64[((22*s0 + 8)//9)]" = split_2[0]
        ne_5: "b8[((22*s0 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_18, -100)
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[((22*s0 + 8)//9)]" = torch.ops.aten.where.self(ne_5, getitem_18, full_default_2)
        unsqueeze: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[((22*s0 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_37, 1, unsqueeze);  sub_37 = unsqueeze = None
        squeeze: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.where.self(ne_5, neg, full_default_3);  ne_5 = neg = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_106: "f32[]" = torch.ops.aten.add.Tensor(sum_3, 0.0);  sum_3 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_1: "bf16[((22*s4 + 8)//9), 262208]" = torch.ops.aten.mm.default(getitem_1, permute_1);  getitem_1 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_6: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32)
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        amax_1: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.amax.default(convert_element_type_6, [1], True)
        sub_42: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_6, amax_1);  convert_element_type_6 = None
        exp_1: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_42)
        sum_4: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [1], True);  exp_1 = None
        log_1: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.log.default(sum_4);  sum_4 = None
        sub_43: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_42, log_1);  sub_42 = None
        getitem_28: "i64[((22*s0 + 8)//9)]" = split_2[1]
        ne_11: "b8[((22*s0 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_28, -100)
        where_2: "i64[((22*s0 + 8)//9)]" = torch.ops.aten.where.self(ne_11, getitem_28, full_default_2)
        unsqueeze_1: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
        gather_1: "f32[((22*s0 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_43, 1, unsqueeze_1);  sub_43 = unsqueeze_1 = None
        squeeze_1: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg_1: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.neg.default(squeeze_1);  squeeze_1 = None
        where_3: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.where.self(ne_11, neg_1, full_default_3);  ne_11 = neg_1 = None
        sum_6: "f32[]" = torch.ops.aten.sum.default(where_3);  where_3 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_119: "f32[]" = torch.ops.aten.add.Tensor(add_106, sum_6);  add_106 = sum_6 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_2: "bf16[((22*s4 + 8)//9), 262208]" = torch.ops.aten.mm.default(getitem_2, permute_1);  getitem_2 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_10: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32)
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        amax_2: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.amax.default(convert_element_type_10, [1], True)
        sub_48: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_10, amax_2);  convert_element_type_10 = None
        exp_2: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_48)
        sum_7: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [1], True);  exp_2 = None
        log_2: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.log.default(sum_7);  sum_7 = None
        sub_49: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_48, log_2);  sub_48 = None
        getitem_38: "i64[((22*s0 + 8)//9)]" = split_2[2]
        ne_17: "b8[((22*s0 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_38, -100)
        where_4: "i64[((22*s0 + 8)//9)]" = torch.ops.aten.where.self(ne_17, getitem_38, full_default_2)
        unsqueeze_2: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where_4, 1);  where_4 = None
        gather_2: "f32[((22*s0 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_49, 1, unsqueeze_2);  sub_49 = unsqueeze_2 = None
        squeeze_2: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather_2, 1);  gather_2 = None
        neg_2: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
        where_5: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.where.self(ne_17, neg_2, full_default_3);  ne_17 = neg_2 = None
        sum_9: "f32[]" = torch.ops.aten.sum.default(where_5);  where_5 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_132: "f32[]" = torch.ops.aten.add.Tensor(add_119, sum_9);  add_119 = sum_9 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_3: "bf16[((22*s4 + 8)//9), 262208]" = torch.ops.aten.mm.default(getitem_3, permute_1);  getitem_3 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_14: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32)
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        amax_3: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.amax.default(convert_element_type_14, [1], True)
        sub_54: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_14, amax_3);  convert_element_type_14 = None
        exp_3: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_54)
        sum_10: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [1], True);  exp_3 = None
        log_3: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.log.default(sum_10);  sum_10 = None
        sub_55: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_54, log_3);  sub_54 = None
        getitem_48: "i64[((22*s0 + 8)//9)]" = split_2[3]
        ne_23: "b8[((22*s0 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_48, -100)
        where_6: "i64[((22*s0 + 8)//9)]" = torch.ops.aten.where.self(ne_23, getitem_48, full_default_2)
        unsqueeze_3: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where_6, 1);  where_6 = None
        gather_3: "f32[((22*s0 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_55, 1, unsqueeze_3);  sub_55 = unsqueeze_3 = None
        squeeze_3: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather_3, 1);  gather_3 = None
        neg_3: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
        where_7: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.where.self(ne_23, neg_3, full_default_3);  ne_23 = neg_3 = None
        sum_12: "f32[]" = torch.ops.aten.sum.default(where_7);  where_7 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_145: "f32[]" = torch.ops.aten.add.Tensor(add_132, sum_12);  add_132 = sum_12 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_4: "bf16[((22*s4 + 8)//9), 262208]" = torch.ops.aten.mm.default(getitem_4, permute_1);  getitem_4 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_18: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32)
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        amax_4: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.amax.default(convert_element_type_18, [1], True)
        sub_60: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_18, amax_4);  convert_element_type_18 = None
        exp_4: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_60)
        sum_13: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [1], True);  exp_4 = None
        log_4: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_61: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_60, log_4);  sub_60 = None
        getitem_58: "i64[((22*s0 + 8)//9)]" = split_2[4]
        ne_29: "b8[((22*s0 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_58, -100)
        where_8: "i64[((22*s0 + 8)//9)]" = torch.ops.aten.where.self(ne_29, getitem_58, full_default_2)
        unsqueeze_4: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where_8, 1);  where_8 = None
        gather_4: "f32[((22*s0 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_61, 1, unsqueeze_4);  sub_61 = unsqueeze_4 = None
        squeeze_4: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather_4, 1);  gather_4 = None
        neg_4: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.neg.default(squeeze_4);  squeeze_4 = None
        where_9: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.where.self(ne_29, neg_4, full_default_3);  ne_29 = neg_4 = None
        sum_15: "f32[]" = torch.ops.aten.sum.default(where_9);  where_9 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_158: "f32[]" = torch.ops.aten.add.Tensor(add_145, sum_15);  add_145 = sum_15 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_5: "bf16[((22*s4 + 8)//9), 262208]" = torch.ops.aten.mm.default(getitem_5, permute_1);  getitem_5 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_22: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32)
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        amax_5: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.amax.default(convert_element_type_22, [1], True)
        sub_66: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_22, amax_5);  convert_element_type_22 = None
        exp_5: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_66)
        sum_16: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [1], True);  exp_5 = None
        log_5: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.log.default(sum_16);  sum_16 = None
        sub_67: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_66, log_5);  sub_66 = None
        getitem_68: "i64[((22*s0 + 8)//9)]" = split_2[5]
        ne_35: "b8[((22*s0 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_68, -100)
        where_10: "i64[((22*s0 + 8)//9)]" = torch.ops.aten.where.self(ne_35, getitem_68, full_default_2)
        unsqueeze_5: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where_10, 1);  where_10 = None
        gather_5: "f32[((22*s0 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_67, 1, unsqueeze_5);  sub_67 = unsqueeze_5 = None
        squeeze_5: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather_5, 1);  gather_5 = None
        neg_5: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.neg.default(squeeze_5);  squeeze_5 = None
        where_11: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.where.self(ne_35, neg_5, full_default_3);  ne_35 = neg_5 = None
        sum_18: "f32[]" = torch.ops.aten.sum.default(where_11);  where_11 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_171: "f32[]" = torch.ops.aten.add.Tensor(add_158, sum_18);  add_158 = sum_18 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_6: "bf16[((22*s4 + 8)//9), 262208]" = torch.ops.aten.mm.default(getitem_6, permute_1);  getitem_6 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_26: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32)
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        amax_6: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.amax.default(convert_element_type_26, [1], True)
        sub_72: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_26, amax_6);  convert_element_type_26 = None
        exp_6: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_72)
        sum_19: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [1], True);  exp_6 = None
        log_6: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.log.default(sum_19);  sum_19 = None
        sub_73: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_72, log_6);  sub_72 = None
        getitem_78: "i64[((22*s0 + 8)//9)]" = split_2[6]
        ne_41: "b8[((22*s0 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_78, -100)
        where_12: "i64[((22*s0 + 8)//9)]" = torch.ops.aten.where.self(ne_41, getitem_78, full_default_2)
        unsqueeze_6: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where_12, 1);  where_12 = None
        gather_6: "f32[((22*s0 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_73, 1, unsqueeze_6);  sub_73 = unsqueeze_6 = None
        squeeze_6: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather_6, 1);  gather_6 = None
        neg_6: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.neg.default(squeeze_6);  squeeze_6 = None
        where_13: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.where.self(ne_41, neg_6, full_default_3);  ne_41 = neg_6 = None
        sum_21: "f32[]" = torch.ops.aten.sum.default(where_13);  where_13 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_184: "f32[]" = torch.ops.aten.add.Tensor(add_171, sum_21);  add_171 = sum_21 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_7: "bf16[((22*s4 + 8)//9), 262208]" = torch.ops.aten.mm.default(getitem_7, permute_1);  getitem_7 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_30: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32)
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        amax_7: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.amax.default(convert_element_type_30, [1], True)
        sub_78: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_30, amax_7);  convert_element_type_30 = None
        exp_7: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_78)
        sum_22: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [1], True);  exp_7 = None
        log_7: "f32[((22*s4 + 8)//9), 1]" = torch.ops.aten.log.default(sum_22);  sum_22 = None
        sub_79: "f32[((22*s4 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_78, log_7);  sub_78 = None
        getitem_88: "i64[((22*s0 + 8)//9)]" = split_2[7]
        ne_47: "b8[((22*s0 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_88, -100)
        where_14: "i64[((22*s0 + 8)//9)]" = torch.ops.aten.where.self(ne_47, getitem_88, full_default_2)
        unsqueeze_7: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where_14, 1);  where_14 = None
        gather_7: "f32[((22*s0 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_79, 1, unsqueeze_7);  sub_79 = unsqueeze_7 = None
        squeeze_7: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather_7, 1);  gather_7 = None
        neg_7: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.neg.default(squeeze_7);  squeeze_7 = None
        where_15: "f32[((22*s0 + 8)//9)]" = torch.ops.aten.where.self(ne_47, neg_7, full_default_3);  ne_47 = neg_7 = None
        sum_24: "f32[]" = torch.ops.aten.sum.default(where_15);  where_15 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_197: "f32[]" = torch.ops.aten.add.Tensor(add_184, sum_24);  add_184 = sum_24 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_8: "bf16[22*s4 - 8*(((22*s4 + 8)//9)), 262208]" = torch.ops.aten.mm.default(getitem_8, permute_1);  getitem_8 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_34: "f32[22*s4 - 8*(((22*s4 + 8)//9)), 262208]" = torch.ops.prims.convert_element_type.default(mm_8, torch.float32)
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        amax_8: "f32[22*s4 - 8*(((22*s4 + 8)//9)), 1]" = torch.ops.aten.amax.default(convert_element_type_34, [1], True)
        sub_84: "f32[22*s4 - 8*(((22*s4 + 8)//9)), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_34, amax_8);  convert_element_type_34 = None
        exp_8: "f32[22*s4 - 8*(((22*s4 + 8)//9)), 262208]" = torch.ops.aten.exp.default(sub_84)
        sum_25: "f32[22*s4 - 8*(((22*s4 + 8)//9)), 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [1], True);  exp_8 = None
        log_8: "f32[22*s4 - 8*(((22*s4 + 8)//9)), 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_85: "f32[22*s4 - 8*(((22*s4 + 8)//9)), 262208]" = torch.ops.aten.sub.Tensor(sub_84, log_8);  sub_84 = None
        getitem_98: "i64[22*s0 - 8*(((22*s0 + 8)//9))]" = split_2[8];  split_2 = None
        ne_53: "b8[22*s0 - 8*(((22*s0 + 8)//9))]" = torch.ops.aten.ne.Scalar(getitem_98, -100)
        where_16: "i64[22*s0 - 8*(((22*s0 + 8)//9))]" = torch.ops.aten.where.self(ne_53, getitem_98, full_default_2)
        unsqueeze_8: "i64[22*s0 - 8*(((22*s0 + 8)//9)), 1]" = torch.ops.aten.unsqueeze.default(where_16, 1);  where_16 = None
        gather_8: "f32[22*s0 - 8*(((22*s0 + 8)//9)), 1]" = torch.ops.aten.gather.default(sub_85, 1, unsqueeze_8);  sub_85 = unsqueeze_8 = None
        squeeze_8: "f32[22*s0 - 8*(((22*s0 + 8)//9))]" = torch.ops.aten.squeeze.dim(gather_8, 1);  gather_8 = None
        neg_8: "f32[22*s0 - 8*(((22*s0 + 8)//9))]" = torch.ops.aten.neg.default(squeeze_8);  squeeze_8 = None
        where_17: "f32[22*s0 - 8*(((22*s0 + 8)//9))]" = torch.ops.aten.where.self(ne_53, neg_8, full_default_3);  ne_53 = neg_8 = full_default_3 = None
        sum_27: "f32[]" = torch.ops.aten.sum.default(where_17);  where_17 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_210: "f32[]" = torch.ops.aten.add.Tensor(add_197, sum_27);  add_197 = sum_27 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:560 in _unsloth_compiled_fused_ce_loss_function, code: loss = loss / torch.tensor(divisor, dtype = torch.float32)
        convert_element_type_36: "f32[]" = torch.ops.prims.convert_element_type.default(primals_8, torch.float32);  primals_8 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(add_210, convert_element_type_36);  add_210 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        unsqueeze_9: "i64[22*s0 - 8*(((22*s0 + 8)//9)), 1]" = torch.ops.aten.unsqueeze.default(getitem_98, 1);  getitem_98 = None
        ne_56: "b8[22*s0 - 8*(((22*s0 + 8)//9)), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_9, -100)
        where_18: "i64[22*s0 - 8*(((22*s0 + 8)//9)), 1]" = torch.ops.aten.where.self(ne_56, unsqueeze_9, full_default_2);  unsqueeze_9 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        permute_10: "bf16[262208, 2560]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        unsqueeze_10: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_88, 1);  getitem_88 = None
        ne_58: "b8[((22*s0 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_10, -100)
        where_20: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_58, unsqueeze_10, full_default_2);  unsqueeze_10 = None
        unsqueeze_11: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_78, 1);  getitem_78 = None
        ne_60: "b8[((22*s0 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_11, -100)
        where_22: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_60, unsqueeze_11, full_default_2);  unsqueeze_11 = None
        unsqueeze_12: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_68, 1);  getitem_68 = None
        ne_62: "b8[((22*s0 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_12, -100)
        where_24: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_62, unsqueeze_12, full_default_2);  unsqueeze_12 = None
        unsqueeze_13: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_58, 1);  getitem_58 = None
        ne_64: "b8[((22*s0 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_13, -100)
        where_26: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_64, unsqueeze_13, full_default_2);  unsqueeze_13 = None
        unsqueeze_14: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_48, 1);  getitem_48 = None
        ne_66: "b8[((22*s0 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_14, -100)
        where_28: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_66, unsqueeze_14, full_default_2);  unsqueeze_14 = None
        unsqueeze_15: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_38, 1);  getitem_38 = None
        ne_68: "b8[((22*s0 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_15, -100)
        where_30: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_68, unsqueeze_15, full_default_2);  unsqueeze_15 = None
        unsqueeze_16: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_28, 1);  getitem_28 = None
        ne_70: "b8[((22*s0 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_16, -100)
        where_32: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_70, unsqueeze_16, full_default_2);  unsqueeze_16 = None
        unsqueeze_17: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_18, 1);  getitem_18 = None
        ne_72: "b8[((22*s0 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_17, -100)
        where_34: "i64[((22*s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_72, unsqueeze_17, full_default_2);  unsqueeze_17 = full_default_2 = None
        return (div, mm, amax, log, mm_1, amax_1, log_1, mm_2, amax_2, log_2, mm_3, amax_3, log_3, mm_4, amax_4, log_4, mm_5, amax_5, log_5, mm_6, amax_6, log_6, mm_7, amax_7, log_7, mm_8, amax_8, log_8, convert_element_type_36, ne_56, where_18, permute_10, ne_58, where_20, ne_60, where_22, ne_62, where_24, ne_64, where_26, ne_66, where_28, ne_68, where_30, ne_70, where_32, ne_72, where_34, primals_6, primals_2, floordiv, sym_size_int_8)
        