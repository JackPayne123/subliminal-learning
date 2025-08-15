
import os
os.environ['PYTORCH_VERSION'] = '2.5.1'
os.environ['PYTORCH_INDEX_URL'] = 'https://download.pytorch.org/whl/cu124'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:[32:256,64:128,256:64,>:32]'
os.environ['TRITON_DISABLE_LINE_INFO'] = '1'
os.environ['TRITON_FRONT_END_DEBUGGING'] = '0'
os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '1'
os.environ['TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_root'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.verbose = False
torch._dynamo.config.recompile_limit = 128
torch._dynamo.config.accumulated_recompile_limit = 128
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.allow_unspec_int_on_nn_module = True
torch._dynamo.config.optimize_ddp = True
torch._dynamo.config.do_not_emit_runtime_asserts = True
torch._dynamo.config.numpy_default_float = 'float32'
torch._dynamo.config.inline_inbuilt_nn_modules = True
torch._dynamo.config.compiled_autograd = False
torch._inductor.config.debug = False
torch._inductor.config.disable_progress = True
torch._inductor.config.verbose_progress = False
torch._inductor.config.dce = True
torch._inductor.config.memory_planning = True
torch._inductor.config.memory_pool = 'none'
torch._inductor.config.epilogue_fusion = True
torch._inductor.config.efficient_conv_bn_eval_fx_passes = True
torch._inductor.config.group_fusion = False
torch._inductor.config.dynamic_scale_rblock = True
torch._inductor.config.max_autotune = False
torch._inductor.config.max_autotune_pointwise = False
torch._inductor.config.max_autotune_gemm = False
torch._inductor.config.max_autotune_gemm_backends = 'ATEN,TRITON,CPP'
torch._inductor.config.autotune_fallback_to_aten = True
torch._inductor.config.autotune_multi_device = True
torch._inductor.config.coordinate_descent_tuning = False
torch._inductor.config.aggressive_fusion = False
torch._inductor.config.combo_kernels = False
torch._inductor.config.benchmark_combo_kernel = False
torch._inductor.config.combo_kernel_foreach_dynamic_shapes = False
torch._inductor.config.emulate_precision_casts = False
torch._inductor.config.compile_threads = 1
torch._inductor.config.shape_padding = True
torch._inductor.config.freezing = False
torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.triton.autotune_at_compile_time = None
torch._inductor.config.triton.cooperative_reductions = False
torch._inductor.config.triton.multi_kernel = 0
torch._inductor.config.triton.store_cubin = False
torch._inductor.config.triton.use_block_ptr = False
torch._inductor.config.triton.enable_persistent_tma_matmul = False
torch._inductor.config.cuda.compile_opt_level = '-O1'
torch._inductor.config.cuda.enable_cuda_lto = True
torch._inductor.config.cuda.use_fast_math = True
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config.trace.graph_diagram = False
torch._inductor.config.test_configs.runtime_triton_dtype_assert = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.7.1+cu126
# torch cuda version: 12.6
# torch git version: e2d141dbde55c2a4370fac5165b0561b6af4798b


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Thu_Mar_28_02:18:24_PDT_2024 
# Cuda compilation tools, release 12.4, V12.4.131 
# Build cuda_12.4.r12.4/compiler.34097967_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 4090 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8):
        empty = torch.ops.aten.empty.memory_format([22, primals_2], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute = torch.ops.aten.permute.default(empty, [0, 1]);  empty = None
        slice_1 = torch.ops.aten.slice.Tensor(primals_3, 1, 1, 9223372036854775807);  primals_3 = None
        slice_scatter = torch.ops.aten.slice_scatter.default(permute, slice_1, 1, 0, -1);  slice_1 = None
        slice_5 = torch.ops.aten.slice.Tensor(primals_5, 1, 1, 9223372036854775807);  primals_5 = None
        eq_16 = torch.ops.aten.eq.Scalar(slice_5, 0);  slice_5 = None
        full_default = torch.ops.aten.full.default([], -100, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        slice_6 = torch.ops.aten.slice.Tensor(slice_scatter, 1, 0, -1)
        index_put = torch.ops.aten.index_put.default(slice_6, [eq_16], full_default);  slice_6 = eq_16 = None
        slice_scatter_1 = torch.ops.aten.slice_scatter.default(slice_scatter, index_put, 1, 0, -1);  slice_scatter = index_put = None
        select_1 = torch.ops.aten.select.int(slice_scatter_1, 1, -1)
        copy_1 = torch.ops.aten.copy.default(select_1, full_default);  select_1 = full_default = None
        select_scatter = torch.ops.aten.select_scatter.default(slice_scatter_1, copy_1, 1, -1);  slice_scatter_1 = copy_1 = None
        sym_numel_default = torch.ops.aten.sym_numel.default(permute);  permute = None
        view_1 = torch.ops.aten.view.default(primals_7, [-1, 2560]);  primals_7 = None
        sym_size_int_7 = torch.ops.aten.sym_size.int(view_1, 0)
        add_47 = sym_size_int_7 + 9;  sym_size_int_7 = None
        sub_14 = add_47 - 1;  add_47 = None
        floordiv = sub_14 // 9;  sub_14 = None
        split = torch.ops.aten.split.Tensor(view_1, floordiv);  view_1 = None
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2]
        getitem_3 = split[3]
        getitem_4 = split[4]
        getitem_5 = split[5]
        getitem_6 = split[6]
        getitem_7 = split[7]
        getitem_8 = split[8];  split = None
        sym_size_int_8 = torch.ops.aten.sym_size.int(getitem_8, 0)
        add_75 = sym_numel_default + 9;  sym_numel_default = None
        sub_24 = add_75 - 1;  add_75 = None
        floordiv_1 = sub_24 // 9;  sub_24 = None
        permute_1 = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        mm = torch.ops.aten.mm.default(getitem, permute_1);  getitem = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(mm, torch.float32)
        amax = torch.ops.aten.amax.default(convert_element_type_2, [1], True)
        sub_36 = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = None
        exp = torch.ops.aten.exp.default(sub_36)
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_37 = torch.ops.aten.sub.Tensor(sub_36, log);  sub_36 = None
        view_2 = torch.ops.aten.view.default(select_scatter, [-1]);  select_scatter = None
        split_2 = torch.ops.aten.split.Tensor(view_2, floordiv_1);  view_2 = floordiv_1 = None
        getitem_18 = split_2[0]
        ne_5 = torch.ops.aten.ne.Scalar(getitem_18, -100)
        full_default_2 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(ne_5, getitem_18, full_default_2)
        unsqueeze = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather = torch.ops.aten.gather.default(sub_37, 1, unsqueeze);  sub_37 = unsqueeze = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_3 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne_5, neg, full_default_3);  ne_5 = neg = None
        sum_3 = torch.ops.aten.sum.default(where_1);  where_1 = None
        add_106 = torch.ops.aten.add.Tensor(sum_3, 0.0);  sum_3 = None
        mm_1 = torch.ops.aten.mm.default(getitem_1, permute_1);  getitem_1 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(mm_1, torch.float32)
        amax_1 = torch.ops.aten.amax.default(convert_element_type_6, [1], True)
        sub_42 = torch.ops.aten.sub.Tensor(convert_element_type_6, amax_1);  convert_element_type_6 = None
        exp_1 = torch.ops.aten.exp.default(sub_42)
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_1, [1], True);  exp_1 = None
        log_1 = torch.ops.aten.log.default(sum_4);  sum_4 = None
        sub_43 = torch.ops.aten.sub.Tensor(sub_42, log_1);  sub_42 = None
        getitem_28 = split_2[1]
        ne_11 = torch.ops.aten.ne.Scalar(getitem_28, -100)
        where_2 = torch.ops.aten.where.self(ne_11, getitem_28, full_default_2)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
        gather_1 = torch.ops.aten.gather.default(sub_43, 1, unsqueeze_1);  sub_43 = unsqueeze_1 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg_1 = torch.ops.aten.neg.default(squeeze_1);  squeeze_1 = None
        where_3 = torch.ops.aten.where.self(ne_11, neg_1, full_default_3);  ne_11 = neg_1 = None
        sum_6 = torch.ops.aten.sum.default(where_3);  where_3 = None
        add_119 = torch.ops.aten.add.Tensor(add_106, sum_6);  add_106 = sum_6 = None
        mm_2 = torch.ops.aten.mm.default(getitem_2, permute_1);  getitem_2 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(mm_2, torch.float32)
        amax_2 = torch.ops.aten.amax.default(convert_element_type_10, [1], True)
        sub_48 = torch.ops.aten.sub.Tensor(convert_element_type_10, amax_2);  convert_element_type_10 = None
        exp_2 = torch.ops.aten.exp.default(sub_48)
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_2, [1], True);  exp_2 = None
        log_2 = torch.ops.aten.log.default(sum_7);  sum_7 = None
        sub_49 = torch.ops.aten.sub.Tensor(sub_48, log_2);  sub_48 = None
        getitem_38 = split_2[2]
        ne_17 = torch.ops.aten.ne.Scalar(getitem_38, -100)
        where_4 = torch.ops.aten.where.self(ne_17, getitem_38, full_default_2)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(where_4, 1);  where_4 = None
        gather_2 = torch.ops.aten.gather.default(sub_49, 1, unsqueeze_2);  sub_49 = unsqueeze_2 = None
        squeeze_2 = torch.ops.aten.squeeze.dim(gather_2, 1);  gather_2 = None
        neg_2 = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
        where_5 = torch.ops.aten.where.self(ne_17, neg_2, full_default_3);  ne_17 = neg_2 = None
        sum_9 = torch.ops.aten.sum.default(where_5);  where_5 = None
        add_132 = torch.ops.aten.add.Tensor(add_119, sum_9);  add_119 = sum_9 = None
        mm_3 = torch.ops.aten.mm.default(getitem_3, permute_1);  getitem_3 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(mm_3, torch.float32)
        amax_3 = torch.ops.aten.amax.default(convert_element_type_14, [1], True)
        sub_54 = torch.ops.aten.sub.Tensor(convert_element_type_14, amax_3);  convert_element_type_14 = None
        exp_3 = torch.ops.aten.exp.default(sub_54)
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_3, [1], True);  exp_3 = None
        log_3 = torch.ops.aten.log.default(sum_10);  sum_10 = None
        sub_55 = torch.ops.aten.sub.Tensor(sub_54, log_3);  sub_54 = None
        getitem_48 = split_2[3]
        ne_23 = torch.ops.aten.ne.Scalar(getitem_48, -100)
        where_6 = torch.ops.aten.where.self(ne_23, getitem_48, full_default_2)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(where_6, 1);  where_6 = None
        gather_3 = torch.ops.aten.gather.default(sub_55, 1, unsqueeze_3);  sub_55 = unsqueeze_3 = None
        squeeze_3 = torch.ops.aten.squeeze.dim(gather_3, 1);  gather_3 = None
        neg_3 = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
        where_7 = torch.ops.aten.where.self(ne_23, neg_3, full_default_3);  ne_23 = neg_3 = None
        sum_12 = torch.ops.aten.sum.default(where_7);  where_7 = None
        add_145 = torch.ops.aten.add.Tensor(add_132, sum_12);  add_132 = sum_12 = None
        mm_4 = torch.ops.aten.mm.default(getitem_4, permute_1);  getitem_4 = None
        convert_element_type_18 = torch.ops.prims.convert_element_type.default(mm_4, torch.float32)
        amax_4 = torch.ops.aten.amax.default(convert_element_type_18, [1], True)
        sub_60 = torch.ops.aten.sub.Tensor(convert_element_type_18, amax_4);  convert_element_type_18 = None
        exp_4 = torch.ops.aten.exp.default(sub_60)
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_4, [1], True);  exp_4 = None
        log_4 = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_61 = torch.ops.aten.sub.Tensor(sub_60, log_4);  sub_60 = None
        getitem_58 = split_2[4]
        ne_29 = torch.ops.aten.ne.Scalar(getitem_58, -100)
        where_8 = torch.ops.aten.where.self(ne_29, getitem_58, full_default_2)
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(where_8, 1);  where_8 = None
        gather_4 = torch.ops.aten.gather.default(sub_61, 1, unsqueeze_4);  sub_61 = unsqueeze_4 = None
        squeeze_4 = torch.ops.aten.squeeze.dim(gather_4, 1);  gather_4 = None
        neg_4 = torch.ops.aten.neg.default(squeeze_4);  squeeze_4 = None
        where_9 = torch.ops.aten.where.self(ne_29, neg_4, full_default_3);  ne_29 = neg_4 = None
        sum_15 = torch.ops.aten.sum.default(where_9);  where_9 = None
        add_158 = torch.ops.aten.add.Tensor(add_145, sum_15);  add_145 = sum_15 = None
        mm_5 = torch.ops.aten.mm.default(getitem_5, permute_1);  getitem_5 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(mm_5, torch.float32)
        amax_5 = torch.ops.aten.amax.default(convert_element_type_22, [1], True)
        sub_66 = torch.ops.aten.sub.Tensor(convert_element_type_22, amax_5);  convert_element_type_22 = None
        exp_5 = torch.ops.aten.exp.default(sub_66)
        sum_16 = torch.ops.aten.sum.dim_IntList(exp_5, [1], True);  exp_5 = None
        log_5 = torch.ops.aten.log.default(sum_16);  sum_16 = None
        sub_67 = torch.ops.aten.sub.Tensor(sub_66, log_5);  sub_66 = None
        getitem_68 = split_2[5]
        ne_35 = torch.ops.aten.ne.Scalar(getitem_68, -100)
        where_10 = torch.ops.aten.where.self(ne_35, getitem_68, full_default_2)
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(where_10, 1);  where_10 = None
        gather_5 = torch.ops.aten.gather.default(sub_67, 1, unsqueeze_5);  sub_67 = unsqueeze_5 = None
        squeeze_5 = torch.ops.aten.squeeze.dim(gather_5, 1);  gather_5 = None
        neg_5 = torch.ops.aten.neg.default(squeeze_5);  squeeze_5 = None
        where_11 = torch.ops.aten.where.self(ne_35, neg_5, full_default_3);  ne_35 = neg_5 = None
        sum_18 = torch.ops.aten.sum.default(where_11);  where_11 = None
        add_171 = torch.ops.aten.add.Tensor(add_158, sum_18);  add_158 = sum_18 = None
        mm_6 = torch.ops.aten.mm.default(getitem_6, permute_1);  getitem_6 = None
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(mm_6, torch.float32)
        amax_6 = torch.ops.aten.amax.default(convert_element_type_26, [1], True)
        sub_72 = torch.ops.aten.sub.Tensor(convert_element_type_26, amax_6);  convert_element_type_26 = None
        exp_6 = torch.ops.aten.exp.default(sub_72)
        sum_19 = torch.ops.aten.sum.dim_IntList(exp_6, [1], True);  exp_6 = None
        log_6 = torch.ops.aten.log.default(sum_19);  sum_19 = None
        sub_73 = torch.ops.aten.sub.Tensor(sub_72, log_6);  sub_72 = None
        getitem_78 = split_2[6]
        ne_41 = torch.ops.aten.ne.Scalar(getitem_78, -100)
        where_12 = torch.ops.aten.where.self(ne_41, getitem_78, full_default_2)
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(where_12, 1);  where_12 = None
        gather_6 = torch.ops.aten.gather.default(sub_73, 1, unsqueeze_6);  sub_73 = unsqueeze_6 = None
        squeeze_6 = torch.ops.aten.squeeze.dim(gather_6, 1);  gather_6 = None
        neg_6 = torch.ops.aten.neg.default(squeeze_6);  squeeze_6 = None
        where_13 = torch.ops.aten.where.self(ne_41, neg_6, full_default_3);  ne_41 = neg_6 = None
        sum_21 = torch.ops.aten.sum.default(where_13);  where_13 = None
        add_184 = torch.ops.aten.add.Tensor(add_171, sum_21);  add_171 = sum_21 = None
        mm_7 = torch.ops.aten.mm.default(getitem_7, permute_1);  getitem_7 = None
        convert_element_type_30 = torch.ops.prims.convert_element_type.default(mm_7, torch.float32)
        amax_7 = torch.ops.aten.amax.default(convert_element_type_30, [1], True)
        sub_78 = torch.ops.aten.sub.Tensor(convert_element_type_30, amax_7);  convert_element_type_30 = None
        exp_7 = torch.ops.aten.exp.default(sub_78)
        sum_22 = torch.ops.aten.sum.dim_IntList(exp_7, [1], True);  exp_7 = None
        log_7 = torch.ops.aten.log.default(sum_22);  sum_22 = None
        sub_79 = torch.ops.aten.sub.Tensor(sub_78, log_7);  sub_78 = None
        getitem_88 = split_2[7]
        ne_47 = torch.ops.aten.ne.Scalar(getitem_88, -100)
        where_14 = torch.ops.aten.where.self(ne_47, getitem_88, full_default_2)
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(where_14, 1);  where_14 = None
        gather_7 = torch.ops.aten.gather.default(sub_79, 1, unsqueeze_7);  sub_79 = unsqueeze_7 = None
        squeeze_7 = torch.ops.aten.squeeze.dim(gather_7, 1);  gather_7 = None
        neg_7 = torch.ops.aten.neg.default(squeeze_7);  squeeze_7 = None
        where_15 = torch.ops.aten.where.self(ne_47, neg_7, full_default_3);  ne_47 = neg_7 = None
        sum_24 = torch.ops.aten.sum.default(where_15);  where_15 = None
        add_197 = torch.ops.aten.add.Tensor(add_184, sum_24);  add_184 = sum_24 = None
        mm_8 = torch.ops.aten.mm.default(getitem_8, permute_1);  getitem_8 = None
        convert_element_type_34 = torch.ops.prims.convert_element_type.default(mm_8, torch.float32)
        amax_8 = torch.ops.aten.amax.default(convert_element_type_34, [1], True)
        sub_84 = torch.ops.aten.sub.Tensor(convert_element_type_34, amax_8);  convert_element_type_34 = None
        exp_8 = torch.ops.aten.exp.default(sub_84)
        sum_25 = torch.ops.aten.sum.dim_IntList(exp_8, [1], True);  exp_8 = None
        log_8 = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_85 = torch.ops.aten.sub.Tensor(sub_84, log_8);  sub_84 = None
        getitem_98 = split_2[8];  split_2 = None
        ne_53 = torch.ops.aten.ne.Scalar(getitem_98, -100)
        where_16 = torch.ops.aten.where.self(ne_53, getitem_98, full_default_2)
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(where_16, 1);  where_16 = None
        gather_8 = torch.ops.aten.gather.default(sub_85, 1, unsqueeze_8);  sub_85 = unsqueeze_8 = None
        squeeze_8 = torch.ops.aten.squeeze.dim(gather_8, 1);  gather_8 = None
        neg_8 = torch.ops.aten.neg.default(squeeze_8);  squeeze_8 = None
        where_17 = torch.ops.aten.where.self(ne_53, neg_8, full_default_3);  ne_53 = neg_8 = full_default_3 = None
        sum_27 = torch.ops.aten.sum.default(where_17);  where_17 = None
        add_210 = torch.ops.aten.add.Tensor(add_197, sum_27);  add_197 = sum_27 = None
        convert_element_type_36 = torch.ops.prims.convert_element_type.default(primals_8, torch.float32);  primals_8 = None
        div = torch.ops.aten.div.Tensor(add_210, convert_element_type_36);  add_210 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(getitem_98, 1);  getitem_98 = None
        ne_56 = torch.ops.aten.ne.Scalar(unsqueeze_9, -100)
        where_18 = torch.ops.aten.where.self(ne_56, unsqueeze_9, full_default_2);  unsqueeze_9 = None
        permute_10 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(getitem_88, 1);  getitem_88 = None
        ne_58 = torch.ops.aten.ne.Scalar(unsqueeze_10, -100)
        where_20 = torch.ops.aten.where.self(ne_58, unsqueeze_10, full_default_2);  unsqueeze_10 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(getitem_78, 1);  getitem_78 = None
        ne_60 = torch.ops.aten.ne.Scalar(unsqueeze_11, -100)
        where_22 = torch.ops.aten.where.self(ne_60, unsqueeze_11, full_default_2);  unsqueeze_11 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(getitem_68, 1);  getitem_68 = None
        ne_62 = torch.ops.aten.ne.Scalar(unsqueeze_12, -100)
        where_24 = torch.ops.aten.where.self(ne_62, unsqueeze_12, full_default_2);  unsqueeze_12 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(getitem_58, 1);  getitem_58 = None
        ne_64 = torch.ops.aten.ne.Scalar(unsqueeze_13, -100)
        where_26 = torch.ops.aten.where.self(ne_64, unsqueeze_13, full_default_2);  unsqueeze_13 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(getitem_48, 1);  getitem_48 = None
        ne_66 = torch.ops.aten.ne.Scalar(unsqueeze_14, -100)
        where_28 = torch.ops.aten.where.self(ne_66, unsqueeze_14, full_default_2);  unsqueeze_14 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(getitem_38, 1);  getitem_38 = None
        ne_68 = torch.ops.aten.ne.Scalar(unsqueeze_15, -100)
        where_30 = torch.ops.aten.where.self(ne_68, unsqueeze_15, full_default_2);  unsqueeze_15 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(getitem_28, 1);  getitem_28 = None
        ne_70 = torch.ops.aten.ne.Scalar(unsqueeze_16, -100)
        where_32 = torch.ops.aten.where.self(ne_70, unsqueeze_16, full_default_2);  unsqueeze_16 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(getitem_18, 1);  getitem_18 = None
        ne_72 = torch.ops.aten.ne.Scalar(unsqueeze_17, -100)
        where_34 = torch.ops.aten.where.self(ne_72, unsqueeze_17, full_default_2);  unsqueeze_17 = full_default_2 = None
        return (div, mm, amax, log, mm_1, amax_1, log_1, mm_2, amax_2, log_2, mm_3, amax_3, log_3, mm_4, amax_4, log_4, mm_5, amax_5, log_5, mm_6, amax_6, log_6, mm_7, amax_7, log_7, mm_8, amax_8, log_8, convert_element_type_36, ne_56, where_18, permute_10, ne_58, where_20, ne_60, where_22, ne_62, where_24, ne_64, where_26, ne_66, where_28, ne_68, where_30, ne_70, where_32, ne_72, where_34, primals_6, primals_2, floordiv, sym_size_int_8)
        
def load_args(reader):
    buf0 = reader.storage(None, 1342504960, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (262208, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    reader.symint(150)  # primals_2
    buf1 = reader.storage(None, 176*s0, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (22, s0), dtype=torch.int64, is_leaf=True)  # primals_3
    reader.symint(150)  # primals_4
    buf2 = reader.storage(None, 176*s1, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf2, (22, s1), dtype=torch.int64, is_leaf=True)  # primals_5
    reader.symint(150)  # primals_6
    buf3 = reader.storage(None, 5120*s3*(s2 - 1) + 5120*s3, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (22, s4, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_7
    buf4 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf4, (), dtype=torch.int64, is_leaf=True)  # primals_8
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='symbolic', check_str=None)
        # mod(*args)