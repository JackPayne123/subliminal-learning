class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[262208, 2560]", primals_2: "Sym(s0)", primals_3: "Sym(s1)", primals_4: "i64[s0, s1]", primals_5: "Sym(s2)", primals_6: "i64[17, s2]", primals_7: "Sym(s5)", primals_8: "Sym(s6)", primals_9: "bf16[s5, s6, 2560]", primals_10: "i64[]"):
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:517 in _unsloth_compiled_fused_ce_loss_function, code: shift_labels = torch.empty_like(output_labels, device = device)
        empty: "i64[s0, s1]" = torch.ops.aten.empty.memory_format([primals_2, primals_3], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute: "i64[s0, s1]" = torch.ops.aten.permute.default(empty, [0, 1]);  empty = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:518 in _unsloth_compiled_fused_ce_loss_function, code: shift_labels[..., :-1] = output_labels[..., 1:]
        slice_1: "i64[s0, s1 - 1]" = torch.ops.aten.slice.Tensor(primals_4, 1, 1, 9223372036854775807);  primals_4 = None
        
        # No stacktrace found for following nodes
        slice_scatter_default: "i64[s0, s1]" = torch.ops.aten.slice_scatter.default(permute, slice_1, 1, 0, -1);  slice_1 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:521 in _unsloth_compiled_fused_ce_loss_function, code: shift_labels[..., :-1][mask[..., 1:] == 0] = -100
        slice_5: "i64[17, s2 - 1]" = torch.ops.aten.slice.Tensor(primals_6, 1, 1, 9223372036854775807);  primals_6 = None
        eq_26: "b8[17, s2 - 1]" = torch.ops.aten.eq.Scalar(slice_5, 0);  slice_5 = None
        full_default: "i64[]" = torch.ops.aten.full.default([], -100, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        slice_6: "i64[s0, s1 - 1]" = torch.ops.aten.slice.Tensor(slice_scatter_default, 1, 0, -1)
        index_put: "i64[s0, s1 - 1]" = torch.ops.aten.index_put_.default(slice_6, [eq_26], full_default);  slice_6 = eq_26 = None
        
        # No stacktrace found for following nodes
        slice_scatter_default_1: "i64[s0, s1]" = torch.ops.aten.slice_scatter.default(slice_scatter_default, index_put, 1, 0, -1);  slice_scatter_default = index_put = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:523 in _unsloth_compiled_fused_ce_loss_function, code: shift_labels[..., -1] = -100
        select_1: "i64[s0]" = torch.ops.aten.select.int(slice_scatter_default_1, 1, -1)
        copy_1: "i64[s0]" = torch.ops.aten.copy.default(select_1, full_default);  select_1 = full_default = None
        
        # No stacktrace found for following nodes
        select_scatter_default: "i64[s0, s1]" = torch.ops.aten.select_scatter.default(slice_scatter_default_1, copy_1, 1, -1);  slice_scatter_default_1 = copy_1 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:524 in _unsloth_compiled_fused_ce_loss_function, code: shift_labels = shift_labels.view(-1)
        sym_numel_default: "Sym(s0*s1)" = torch.ops.aten.sym_numel.default(permute);  permute = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:528 in _unsloth_compiled_fused_ce_loss_function, code: hidden_states = hidden_states.view(-1, hd)
        view_1: "bf16[s5*s6, 2560]" = torch.ops.aten.reshape.default(primals_9, [-1, 2560]);  primals_9 = None
        sym_size_int_8: "Sym(s5*s6)" = torch.ops.aten.sym_size.int(view_1, 0)
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:530 in _unsloth_compiled_fused_ce_loss_function, code: __shift_states = torch.chunk(hidden_states, n_chunks, dim = 0)
        add_47: "Sym(s5*s6 + 9)" = sym_size_int_8 + 9;  sym_size_int_8 = None
        sub_26: "Sym(s5*s6 + 8)" = add_47 - 1;  add_47 = None
        floordiv: "Sym(((s5*s6 + 8)//9))" = sub_26 // 9;  sub_26 = None
        split = torch.ops.aten.split.Tensor(view_1, floordiv);  view_1 = None
        getitem: "bf16[((s5*s6 + 8)//9), 2560]" = split[0]
        getitem_1: "bf16[((s5*s6 + 8)//9), 2560]" = split[1]
        getitem_2: "bf16[((s5*s6 + 8)//9), 2560]" = split[2]
        getitem_3: "bf16[((s5*s6 + 8)//9), 2560]" = split[3]
        getitem_4: "bf16[((s5*s6 + 8)//9), 2560]" = split[4]
        getitem_5: "bf16[((s5*s6 + 8)//9), 2560]" = split[5]
        getitem_6: "bf16[((s5*s6 + 8)//9), 2560]" = split[6]
        getitem_7: "bf16[((s5*s6 + 8)//9), 2560]" = split[7]
        getitem_8: "bf16[s5*s6 - 8*(((s5*s6 + 8)//9)), 2560]" = split[8];  split = None
        sym_size_int_9: "Sym(s5*s6 - 8*(((s5*s6 + 8)//9)))" = torch.ops.aten.sym_size.int(getitem_8, 0)
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:531 in _unsloth_compiled_fused_ce_loss_function, code: __shift_labels = torch.chunk(shift_labels,  n_chunks, dim = 0)
        add_75: "Sym(s0*s1 + 9)" = sym_numel_default + 9;  sym_numel_default = None
        sub_36: "Sym(s0*s1 + 8)" = add_75 - 1;  add_75 = None
        floordiv_1: "Sym(((s0*s1 + 8)//9))" = sub_36 // 9;  sub_36 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        permute_1: "bf16[2560, 262208]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        mm: "bf16[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mm.default(getitem, permute_1);  getitem = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_2: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm, torch.float32)
        
        # No stacktrace found for following nodes
        prepare_softmax_online_default_8 = torch.ops.prims.prepare_softmax_online.default(convert_element_type_2, 1)
        getitem_115: "f32[((s5*s6 + 8)//9), 1]" = prepare_softmax_online_default_8[0]
        getitem_116: "f32[((s5*s6 + 8)//9), 1]" = prepare_softmax_online_default_8[1];  prepare_softmax_online_default_8 = None
        sub_tensor_8: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_2, getitem_115);  convert_element_type_2 = None
        exp_default_8: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_tensor_8);  exp_default_8 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        log: "f32[((s5*s6 + 8)//9), 1]" = torch.ops.aten.log.default(getitem_116);  getitem_116 = None
        sub_49: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_tensor_8, log);  sub_tensor_8 = None
        view_2: "i64[s0*s1]" = torch.ops.aten.reshape.default(select_scatter_default, [-1]);  select_scatter_default = None
        split_2 = torch.ops.aten.split.Tensor(view_2, floordiv_1);  view_2 = floordiv_1 = None
        getitem_18: "i64[((s0*s1 + 8)//9)]" = split_2[0]
        ne_6: "b8[((s0*s1 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_18, -100)
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[((s0*s1 + 8)//9)]" = torch.ops.aten.where.self(ne_6, getitem_18, full_default_2)
        unsqueeze: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[((s0*s1 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_49, 1, unsqueeze);  sub_49 = unsqueeze = None
        squeeze: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.where.self(ne_6, neg, full_default_3);  ne_6 = neg = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_106: "f32[]" = torch.ops.aten.add.Tensor(sum_3, 0.0);  sum_3 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_1: "bf16[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mm.default(getitem_1, permute_1);  getitem_1 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_6: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32)
        
        # No stacktrace found for following nodes
        prepare_softmax_online_default_7 = torch.ops.prims.prepare_softmax_online.default(convert_element_type_6, 1)
        getitem_113: "f32[((s5*s6 + 8)//9), 1]" = prepare_softmax_online_default_7[0]
        getitem_114: "f32[((s5*s6 + 8)//9), 1]" = prepare_softmax_online_default_7[1];  prepare_softmax_online_default_7 = None
        sub_tensor_7: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_6, getitem_113);  convert_element_type_6 = None
        exp_default_7: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_tensor_7);  exp_default_7 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        log_1: "f32[((s5*s6 + 8)//9), 1]" = torch.ops.aten.log.default(getitem_114);  getitem_114 = None
        sub_55: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_tensor_7, log_1);  sub_tensor_7 = None
        getitem_28: "i64[((s0*s1 + 8)//9)]" = split_2[1]
        ne_12: "b8[((s0*s1 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_28, -100)
        where_2: "i64[((s0*s1 + 8)//9)]" = torch.ops.aten.where.self(ne_12, getitem_28, full_default_2)
        unsqueeze_1: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
        gather_1: "f32[((s0*s1 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_55, 1, unsqueeze_1);  sub_55 = unsqueeze_1 = None
        squeeze_1: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg_1: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.neg.default(squeeze_1);  squeeze_1 = None
        where_3: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.where.self(ne_12, neg_1, full_default_3);  ne_12 = neg_1 = None
        sum_6: "f32[]" = torch.ops.aten.sum.default(where_3);  where_3 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_119: "f32[]" = torch.ops.aten.add.Tensor(add_106, sum_6);  add_106 = sum_6 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_2: "bf16[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mm.default(getitem_2, permute_1);  getitem_2 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_10: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32)
        
        # No stacktrace found for following nodes
        prepare_softmax_online_default_6 = torch.ops.prims.prepare_softmax_online.default(convert_element_type_10, 1)
        getitem_111: "f32[((s5*s6 + 8)//9), 1]" = prepare_softmax_online_default_6[0]
        getitem_112: "f32[((s5*s6 + 8)//9), 1]" = prepare_softmax_online_default_6[1];  prepare_softmax_online_default_6 = None
        sub_tensor_6: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_10, getitem_111);  convert_element_type_10 = None
        exp_default_6: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_tensor_6);  exp_default_6 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        log_2: "f32[((s5*s6 + 8)//9), 1]" = torch.ops.aten.log.default(getitem_112);  getitem_112 = None
        sub_61: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_tensor_6, log_2);  sub_tensor_6 = None
        getitem_38: "i64[((s0*s1 + 8)//9)]" = split_2[2]
        ne_18: "b8[((s0*s1 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_38, -100)
        where_4: "i64[((s0*s1 + 8)//9)]" = torch.ops.aten.where.self(ne_18, getitem_38, full_default_2)
        unsqueeze_2: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where_4, 1);  where_4 = None
        gather_2: "f32[((s0*s1 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_61, 1, unsqueeze_2);  sub_61 = unsqueeze_2 = None
        squeeze_2: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather_2, 1);  gather_2 = None
        neg_2: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
        where_5: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.where.self(ne_18, neg_2, full_default_3);  ne_18 = neg_2 = None
        sum_9: "f32[]" = torch.ops.aten.sum.default(where_5);  where_5 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_132: "f32[]" = torch.ops.aten.add.Tensor(add_119, sum_9);  add_119 = sum_9 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_3: "bf16[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mm.default(getitem_3, permute_1);  getitem_3 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_14: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32)
        
        # No stacktrace found for following nodes
        prepare_softmax_online_default_5 = torch.ops.prims.prepare_softmax_online.default(convert_element_type_14, 1)
        getitem_109: "f32[((s5*s6 + 8)//9), 1]" = prepare_softmax_online_default_5[0]
        getitem_110: "f32[((s5*s6 + 8)//9), 1]" = prepare_softmax_online_default_5[1];  prepare_softmax_online_default_5 = None
        sub_tensor_5: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_14, getitem_109);  convert_element_type_14 = None
        exp_default_5: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_tensor_5);  exp_default_5 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        log_3: "f32[((s5*s6 + 8)//9), 1]" = torch.ops.aten.log.default(getitem_110);  getitem_110 = None
        sub_67: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_tensor_5, log_3);  sub_tensor_5 = None
        getitem_48: "i64[((s0*s1 + 8)//9)]" = split_2[3]
        ne_24: "b8[((s0*s1 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_48, -100)
        where_6: "i64[((s0*s1 + 8)//9)]" = torch.ops.aten.where.self(ne_24, getitem_48, full_default_2)
        unsqueeze_3: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where_6, 1);  where_6 = None
        gather_3: "f32[((s0*s1 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_67, 1, unsqueeze_3);  sub_67 = unsqueeze_3 = None
        squeeze_3: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather_3, 1);  gather_3 = None
        neg_3: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
        where_7: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.where.self(ne_24, neg_3, full_default_3);  ne_24 = neg_3 = None
        sum_12: "f32[]" = torch.ops.aten.sum.default(where_7);  where_7 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_145: "f32[]" = torch.ops.aten.add.Tensor(add_132, sum_12);  add_132 = sum_12 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_4: "bf16[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mm.default(getitem_4, permute_1);  getitem_4 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_18: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32)
        
        # No stacktrace found for following nodes
        prepare_softmax_online_default_4 = torch.ops.prims.prepare_softmax_online.default(convert_element_type_18, 1)
        getitem_107: "f32[((s5*s6 + 8)//9), 1]" = prepare_softmax_online_default_4[0]
        getitem_108: "f32[((s5*s6 + 8)//9), 1]" = prepare_softmax_online_default_4[1];  prepare_softmax_online_default_4 = None
        sub_tensor_4: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_18, getitem_107);  convert_element_type_18 = None
        exp_default_4: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_tensor_4);  exp_default_4 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        log_4: "f32[((s5*s6 + 8)//9), 1]" = torch.ops.aten.log.default(getitem_108);  getitem_108 = None
        sub_73: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_tensor_4, log_4);  sub_tensor_4 = None
        getitem_58: "i64[((s0*s1 + 8)//9)]" = split_2[4]
        ne_30: "b8[((s0*s1 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_58, -100)
        where_8: "i64[((s0*s1 + 8)//9)]" = torch.ops.aten.where.self(ne_30, getitem_58, full_default_2)
        unsqueeze_4: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where_8, 1);  where_8 = None
        gather_4: "f32[((s0*s1 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_73, 1, unsqueeze_4);  sub_73 = unsqueeze_4 = None
        squeeze_4: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather_4, 1);  gather_4 = None
        neg_4: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.neg.default(squeeze_4);  squeeze_4 = None
        where_9: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.where.self(ne_30, neg_4, full_default_3);  ne_30 = neg_4 = None
        sum_15: "f32[]" = torch.ops.aten.sum.default(where_9);  where_9 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_158: "f32[]" = torch.ops.aten.add.Tensor(add_145, sum_15);  add_145 = sum_15 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_5: "bf16[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mm.default(getitem_5, permute_1);  getitem_5 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_22: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32)
        
        # No stacktrace found for following nodes
        prepare_softmax_online_default_3 = torch.ops.prims.prepare_softmax_online.default(convert_element_type_22, 1)
        getitem_105: "f32[((s5*s6 + 8)//9), 1]" = prepare_softmax_online_default_3[0]
        getitem_106: "f32[((s5*s6 + 8)//9), 1]" = prepare_softmax_online_default_3[1];  prepare_softmax_online_default_3 = None
        sub_tensor_3: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_22, getitem_105);  convert_element_type_22 = None
        exp_default_3: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_tensor_3);  exp_default_3 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        log_5: "f32[((s5*s6 + 8)//9), 1]" = torch.ops.aten.log.default(getitem_106);  getitem_106 = None
        sub_79: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_tensor_3, log_5);  sub_tensor_3 = None
        getitem_68: "i64[((s0*s1 + 8)//9)]" = split_2[5]
        ne_36: "b8[((s0*s1 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_68, -100)
        where_10: "i64[((s0*s1 + 8)//9)]" = torch.ops.aten.where.self(ne_36, getitem_68, full_default_2)
        unsqueeze_5: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where_10, 1);  where_10 = None
        gather_5: "f32[((s0*s1 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_79, 1, unsqueeze_5);  sub_79 = unsqueeze_5 = None
        squeeze_5: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather_5, 1);  gather_5 = None
        neg_5: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.neg.default(squeeze_5);  squeeze_5 = None
        where_11: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.where.self(ne_36, neg_5, full_default_3);  ne_36 = neg_5 = None
        sum_18: "f32[]" = torch.ops.aten.sum.default(where_11);  where_11 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_171: "f32[]" = torch.ops.aten.add.Tensor(add_158, sum_18);  add_158 = sum_18 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_6: "bf16[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mm.default(getitem_6, permute_1);  getitem_6 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_26: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32)
        
        # No stacktrace found for following nodes
        prepare_softmax_online_default_2 = torch.ops.prims.prepare_softmax_online.default(convert_element_type_26, 1)
        getitem_103: "f32[((s5*s6 + 8)//9), 1]" = prepare_softmax_online_default_2[0]
        getitem_104: "f32[((s5*s6 + 8)//9), 1]" = prepare_softmax_online_default_2[1];  prepare_softmax_online_default_2 = None
        sub_tensor_2: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_26, getitem_103);  convert_element_type_26 = None
        exp_default_2: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_tensor_2);  exp_default_2 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        log_6: "f32[((s5*s6 + 8)//9), 1]" = torch.ops.aten.log.default(getitem_104);  getitem_104 = None
        sub_85: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_tensor_2, log_6);  sub_tensor_2 = None
        getitem_78: "i64[((s0*s1 + 8)//9)]" = split_2[6]
        ne_42: "b8[((s0*s1 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_78, -100)
        where_12: "i64[((s0*s1 + 8)//9)]" = torch.ops.aten.where.self(ne_42, getitem_78, full_default_2)
        unsqueeze_6: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where_12, 1);  where_12 = None
        gather_6: "f32[((s0*s1 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_85, 1, unsqueeze_6);  sub_85 = unsqueeze_6 = None
        squeeze_6: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather_6, 1);  gather_6 = None
        neg_6: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.neg.default(squeeze_6);  squeeze_6 = None
        where_13: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.where.self(ne_42, neg_6, full_default_3);  ne_42 = neg_6 = None
        sum_21: "f32[]" = torch.ops.aten.sum.default(where_13);  where_13 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_184: "f32[]" = torch.ops.aten.add.Tensor(add_171, sum_21);  add_171 = sum_21 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_7: "bf16[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.mm.default(getitem_7, permute_1);  getitem_7 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_30: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32)
        
        # No stacktrace found for following nodes
        prepare_softmax_online_default_1 = torch.ops.prims.prepare_softmax_online.default(convert_element_type_30, 1)
        getitem_101: "f32[((s5*s6 + 8)//9), 1]" = prepare_softmax_online_default_1[0]
        getitem_102: "f32[((s5*s6 + 8)//9), 1]" = prepare_softmax_online_default_1[1];  prepare_softmax_online_default_1 = None
        sub_tensor_1: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_30, getitem_101);  convert_element_type_30 = None
        exp_default_1: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.exp.default(sub_tensor_1);  exp_default_1 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        log_7: "f32[((s5*s6 + 8)//9), 1]" = torch.ops.aten.log.default(getitem_102);  getitem_102 = None
        sub_91: "f32[((s5*s6 + 8)//9), 262208]" = torch.ops.aten.sub.Tensor(sub_tensor_1, log_7);  sub_tensor_1 = None
        getitem_88: "i64[((s0*s1 + 8)//9)]" = split_2[7]
        ne_48: "b8[((s0*s1 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_88, -100)
        where_14: "i64[((s0*s1 + 8)//9)]" = torch.ops.aten.where.self(ne_48, getitem_88, full_default_2)
        unsqueeze_7: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where_14, 1);  where_14 = None
        gather_7: "f32[((s0*s1 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_91, 1, unsqueeze_7);  sub_91 = unsqueeze_7 = None
        squeeze_7: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather_7, 1);  gather_7 = None
        neg_7: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.neg.default(squeeze_7);  squeeze_7 = None
        where_15: "f32[((s0*s1 + 8)//9)]" = torch.ops.aten.where.self(ne_48, neg_7, full_default_3);  ne_48 = neg_7 = None
        sum_24: "f32[]" = torch.ops.aten.sum.default(where_15);  where_15 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_197: "f32[]" = torch.ops.aten.add.Tensor(add_184, sum_24);  add_184 = sum_24 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_8: "bf16[s5*s6 - 8*(((s5*s6 + 8)//9)), 262208]" = torch.ops.aten.mm.default(getitem_8, permute_1);  getitem_8 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_34: "f32[s5*s6 - 8*(((s5*s6 + 8)//9)), 262208]" = torch.ops.prims.convert_element_type.default(mm_8, torch.float32)
        
        # No stacktrace found for following nodes
        prepare_softmax_online_default = torch.ops.prims.prepare_softmax_online.default(convert_element_type_34, 1)
        getitem_99: "f32[s5*s6 - 8*(((s5*s6 + 8)//9)), 1]" = prepare_softmax_online_default[0]
        getitem_100: "f32[s5*s6 - 8*(((s5*s6 + 8)//9)), 1]" = prepare_softmax_online_default[1];  prepare_softmax_online_default = None
        sub_tensor: "f32[s5*s6 - 8*(((s5*s6 + 8)//9)), 262208]" = torch.ops.aten.sub.Tensor(convert_element_type_34, getitem_99);  convert_element_type_34 = None
        exp_default: "f32[s5*s6 - 8*(((s5*s6 + 8)//9)), 262208]" = torch.ops.aten.exp.default(sub_tensor);  exp_default = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        log_8: "f32[s5*s6 - 8*(((s5*s6 + 8)//9)), 1]" = torch.ops.aten.log.default(getitem_100);  getitem_100 = None
        sub_97: "f32[s5*s6 - 8*(((s5*s6 + 8)//9)), 262208]" = torch.ops.aten.sub.Tensor(sub_tensor, log_8);  sub_tensor = None
        getitem_98: "i64[s0*s1 - 8*(((s0*s1 + 8)//9))]" = split_2[8];  split_2 = None
        ne_54: "b8[s0*s1 - 8*(((s0*s1 + 8)//9))]" = torch.ops.aten.ne.Scalar(getitem_98, -100)
        where_16: "i64[s0*s1 - 8*(((s0*s1 + 8)//9))]" = torch.ops.aten.where.self(ne_54, getitem_98, full_default_2)
        unsqueeze_8: "i64[s0*s1 - 8*(((s0*s1 + 8)//9)), 1]" = torch.ops.aten.unsqueeze.default(where_16, 1);  where_16 = None
        gather_8: "f32[s0*s1 - 8*(((s0*s1 + 8)//9)), 1]" = torch.ops.aten.gather.default(sub_97, 1, unsqueeze_8);  sub_97 = unsqueeze_8 = None
        squeeze_8: "f32[s0*s1 - 8*(((s0*s1 + 8)//9))]" = torch.ops.aten.squeeze.dim(gather_8, 1);  gather_8 = None
        neg_8: "f32[s0*s1 - 8*(((s0*s1 + 8)//9))]" = torch.ops.aten.neg.default(squeeze_8);  squeeze_8 = None
        where_17: "f32[s0*s1 - 8*(((s0*s1 + 8)//9))]" = torch.ops.aten.where.self(ne_54, neg_8, full_default_3);  ne_54 = neg_8 = full_default_3 = None
        sum_27: "f32[]" = torch.ops.aten.sum.default(where_17);  where_17 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_210: "f32[]" = torch.ops.aten.add.Tensor(add_197, sum_27);  add_197 = sum_27 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:560 in _unsloth_compiled_fused_ce_loss_function, code: loss = loss / torch.tensor(divisor, dtype = torch.float32)
        convert_element_type_36: "f32[]" = torch.ops.prims.convert_element_type.default(primals_10, torch.float32);  primals_10 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(add_210, convert_element_type_36);  add_210 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        unsqueeze_9: "i64[s0*s1 - 8*(((s0*s1 + 8)//9)), 1]" = torch.ops.aten.unsqueeze.default(getitem_98, 1);  getitem_98 = None
        ne_57: "b8[s0*s1 - 8*(((s0*s1 + 8)//9)), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_9, -100)
        where_18: "i64[s0*s1 - 8*(((s0*s1 + 8)//9)), 1]" = torch.ops.aten.where.self(ne_57, unsqueeze_9, full_default_2);  unsqueeze_9 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        permute_10: "bf16[262208, 2560]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        
         # File: /workspace/subliminal-learning/.venv/lib/python3.11/site-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        unsqueeze_10: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_88, 1);  getitem_88 = None
        ne_59: "b8[((s0*s1 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_10, -100)
        where_20: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.where.self(ne_59, unsqueeze_10, full_default_2);  unsqueeze_10 = None
        unsqueeze_11: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_78, 1);  getitem_78 = None
        ne_61: "b8[((s0*s1 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_11, -100)
        where_22: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.where.self(ne_61, unsqueeze_11, full_default_2);  unsqueeze_11 = None
        unsqueeze_12: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_68, 1);  getitem_68 = None
        ne_63: "b8[((s0*s1 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_12, -100)
        where_24: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.where.self(ne_63, unsqueeze_12, full_default_2);  unsqueeze_12 = None
        unsqueeze_13: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_58, 1);  getitem_58 = None
        ne_65: "b8[((s0*s1 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_13, -100)
        where_26: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.where.self(ne_65, unsqueeze_13, full_default_2);  unsqueeze_13 = None
        unsqueeze_14: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_48, 1);  getitem_48 = None
        ne_67: "b8[((s0*s1 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_14, -100)
        where_28: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.where.self(ne_67, unsqueeze_14, full_default_2);  unsqueeze_14 = None
        unsqueeze_15: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_38, 1);  getitem_38 = None
        ne_69: "b8[((s0*s1 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_15, -100)
        where_30: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.where.self(ne_69, unsqueeze_15, full_default_2);  unsqueeze_15 = None
        unsqueeze_16: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_28, 1);  getitem_28 = None
        ne_71: "b8[((s0*s1 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_16, -100)
        where_32: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.where.self(ne_71, unsqueeze_16, full_default_2);  unsqueeze_16 = None
        unsqueeze_17: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_18, 1);  getitem_18 = None
        ne_73: "b8[((s0*s1 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_17, -100)
        where_34: "i64[((s0*s1 + 8)//9), 1]" = torch.ops.aten.where.self(ne_73, unsqueeze_17, full_default_2);  unsqueeze_17 = full_default_2 = None
        return (div, mm, getitem_115, log, mm_1, getitem_113, log_1, mm_2, getitem_111, log_2, mm_3, getitem_109, log_3, mm_4, getitem_107, log_4, mm_5, getitem_105, log_5, mm_6, getitem_103, log_6, mm_7, getitem_101, log_7, mm_8, getitem_99, log_8, convert_element_type_36, ne_57, where_18, permute_10, ne_59, where_20, ne_61, where_22, ne_63, where_24, ne_65, where_26, ne_67, where_28, ne_69, where_30, ne_71, where_32, ne_73, where_34, primals_7, primals_8, primals_2, primals_3, floordiv, sym_size_int_9)
        