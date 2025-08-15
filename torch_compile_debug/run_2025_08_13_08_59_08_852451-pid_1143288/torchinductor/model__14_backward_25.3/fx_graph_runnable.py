
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
os.environ['TRITON_CACHE_DIR'] = '/tmp/torchinductor_root/triton/0'

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

    
    
    def forward(self, primals_7, primals_8, primals_2, primals_3, floordiv, sym_size_int_9, mm, amax, log, mm_1, amax_1, log_1, mm_2, amax_2, log_2, mm_3, amax_3, log_3, mm_4, amax_4, log_4, mm_5, amax_5, log_5, mm_6, amax_6, log_6, mm_7, amax_7, log_7, mm_8, amax_8, log_8, convert_element_type_36, ne_57, where_18, permute_10, ne_59, where_20, ne_61, where_22, ne_63, where_24, ne_65, where_26, ne_67, where_28, ne_69, where_30, ne_71, where_32, ne_73, where_34, tangents_1):
        div_1 = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_36);  tangents_1 = convert_element_type_36 = None
        full = torch.ops.aten.full.default([sym_size_int_9, 262208], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  sym_size_int_9 = None
        scatter = torch.ops.aten.scatter.value(full, 1, where_18, -1.0);  full = where_18 = None
        full_default_3 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19 = torch.ops.aten.where.self(ne_57, div_1, full_default_3);  ne_57 = None
        mul_149 = torch.ops.aten.mul.Tensor(scatter, where_19);  scatter = where_19 = None
        convert_element_type_34 = torch.ops.prims.convert_element_type.default(mm_8, torch.float32);  mm_8 = None
        sub_96 = torch.ops.aten.sub.Tensor(convert_element_type_34, amax_8);  convert_element_type_34 = amax_8 = None
        sub_97 = torch.ops.aten.sub.Tensor(sub_96, log_8);  sub_96 = log_8 = None
        exp_9 = torch.ops.aten.exp.default(sub_97);  sub_97 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(mul_149, [1], True)
        mul_150 = torch.ops.aten.mul.Tensor(exp_9, sum_28);  exp_9 = sum_28 = None
        sub_100 = torch.ops.aten.sub.Tensor(mul_149, mul_150);  mul_149 = mul_150 = None
        convert_element_type_37 = torch.ops.prims.convert_element_type.default(sub_100, torch.bfloat16);  sub_100 = None
        mm_9 = torch.ops.aten.mm.default(convert_element_type_37, permute_10);  convert_element_type_37 = None
        full_1 = torch.ops.aten.full.default([floordiv, 262208], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  floordiv = None
        scatter_1 = torch.ops.aten.scatter.value(full_1, 1, where_20, -1.0);  where_20 = None
        where_21 = torch.ops.aten.where.self(ne_59, div_1, full_default_3);  ne_59 = None
        mul_151 = torch.ops.aten.mul.Tensor(scatter_1, where_21);  scatter_1 = where_21 = None
        convert_element_type_30 = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        sub_90 = torch.ops.aten.sub.Tensor(convert_element_type_30, amax_7);  convert_element_type_30 = amax_7 = None
        sub_91 = torch.ops.aten.sub.Tensor(sub_90, log_7);  sub_90 = log_7 = None
        exp_10 = torch.ops.aten.exp.default(sub_91);  sub_91 = None
        sum_29 = torch.ops.aten.sum.dim_IntList(mul_151, [1], True)
        mul_152 = torch.ops.aten.mul.Tensor(exp_10, sum_29);  exp_10 = sum_29 = None
        sub_101 = torch.ops.aten.sub.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
        convert_element_type_40 = torch.ops.prims.convert_element_type.default(sub_101, torch.bfloat16);  sub_101 = None
        mm_10 = torch.ops.aten.mm.default(convert_element_type_40, permute_10);  convert_element_type_40 = None
        scatter_2 = torch.ops.aten.scatter.value(full_1, 1, where_22, -1.0);  where_22 = None
        where_23 = torch.ops.aten.where.self(ne_61, div_1, full_default_3);  ne_61 = None
        mul_153 = torch.ops.aten.mul.Tensor(scatter_2, where_23);  scatter_2 = where_23 = None
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        sub_84 = torch.ops.aten.sub.Tensor(convert_element_type_26, amax_6);  convert_element_type_26 = amax_6 = None
        sub_85 = torch.ops.aten.sub.Tensor(sub_84, log_6);  sub_84 = log_6 = None
        exp_11 = torch.ops.aten.exp.default(sub_85);  sub_85 = None
        sum_30 = torch.ops.aten.sum.dim_IntList(mul_153, [1], True)
        mul_154 = torch.ops.aten.mul.Tensor(exp_11, sum_30);  exp_11 = sum_30 = None
        sub_102 = torch.ops.aten.sub.Tensor(mul_153, mul_154);  mul_153 = mul_154 = None
        convert_element_type_43 = torch.ops.prims.convert_element_type.default(sub_102, torch.bfloat16);  sub_102 = None
        mm_11 = torch.ops.aten.mm.default(convert_element_type_43, permute_10);  convert_element_type_43 = None
        scatter_3 = torch.ops.aten.scatter.value(full_1, 1, where_24, -1.0);  where_24 = None
        where_25 = torch.ops.aten.where.self(ne_63, div_1, full_default_3);  ne_63 = None
        mul_155 = torch.ops.aten.mul.Tensor(scatter_3, where_25);  scatter_3 = where_25 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        sub_78 = torch.ops.aten.sub.Tensor(convert_element_type_22, amax_5);  convert_element_type_22 = amax_5 = None
        sub_79 = torch.ops.aten.sub.Tensor(sub_78, log_5);  sub_78 = log_5 = None
        exp_12 = torch.ops.aten.exp.default(sub_79);  sub_79 = None
        sum_31 = torch.ops.aten.sum.dim_IntList(mul_155, [1], True)
        mul_156 = torch.ops.aten.mul.Tensor(exp_12, sum_31);  exp_12 = sum_31 = None
        sub_103 = torch.ops.aten.sub.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
        convert_element_type_46 = torch.ops.prims.convert_element_type.default(sub_103, torch.bfloat16);  sub_103 = None
        mm_12 = torch.ops.aten.mm.default(convert_element_type_46, permute_10);  convert_element_type_46 = None
        scatter_4 = torch.ops.aten.scatter.value(full_1, 1, where_26, -1.0);  where_26 = None
        where_27 = torch.ops.aten.where.self(ne_65, div_1, full_default_3);  ne_65 = None
        mul_157 = torch.ops.aten.mul.Tensor(scatter_4, where_27);  scatter_4 = where_27 = None
        convert_element_type_18 = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        sub_72 = torch.ops.aten.sub.Tensor(convert_element_type_18, amax_4);  convert_element_type_18 = amax_4 = None
        sub_73 = torch.ops.aten.sub.Tensor(sub_72, log_4);  sub_72 = log_4 = None
        exp_13 = torch.ops.aten.exp.default(sub_73);  sub_73 = None
        sum_32 = torch.ops.aten.sum.dim_IntList(mul_157, [1], True)
        mul_158 = torch.ops.aten.mul.Tensor(exp_13, sum_32);  exp_13 = sum_32 = None
        sub_104 = torch.ops.aten.sub.Tensor(mul_157, mul_158);  mul_157 = mul_158 = None
        convert_element_type_49 = torch.ops.prims.convert_element_type.default(sub_104, torch.bfloat16);  sub_104 = None
        mm_13 = torch.ops.aten.mm.default(convert_element_type_49, permute_10);  convert_element_type_49 = None
        scatter_5 = torch.ops.aten.scatter.value(full_1, 1, where_28, -1.0);  where_28 = None
        where_29 = torch.ops.aten.where.self(ne_67, div_1, full_default_3);  ne_67 = None
        mul_159 = torch.ops.aten.mul.Tensor(scatter_5, where_29);  scatter_5 = where_29 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        sub_66 = torch.ops.aten.sub.Tensor(convert_element_type_14, amax_3);  convert_element_type_14 = amax_3 = None
        sub_67 = torch.ops.aten.sub.Tensor(sub_66, log_3);  sub_66 = log_3 = None
        exp_14 = torch.ops.aten.exp.default(sub_67);  sub_67 = None
        sum_33 = torch.ops.aten.sum.dim_IntList(mul_159, [1], True)
        mul_160 = torch.ops.aten.mul.Tensor(exp_14, sum_33);  exp_14 = sum_33 = None
        sub_105 = torch.ops.aten.sub.Tensor(mul_159, mul_160);  mul_159 = mul_160 = None
        convert_element_type_52 = torch.ops.prims.convert_element_type.default(sub_105, torch.bfloat16);  sub_105 = None
        mm_14 = torch.ops.aten.mm.default(convert_element_type_52, permute_10);  convert_element_type_52 = None
        scatter_6 = torch.ops.aten.scatter.value(full_1, 1, where_30, -1.0);  where_30 = None
        where_31 = torch.ops.aten.where.self(ne_69, div_1, full_default_3);  ne_69 = None
        mul_161 = torch.ops.aten.mul.Tensor(scatter_6, where_31);  scatter_6 = where_31 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        sub_60 = torch.ops.aten.sub.Tensor(convert_element_type_10, amax_2);  convert_element_type_10 = amax_2 = None
        sub_61 = torch.ops.aten.sub.Tensor(sub_60, log_2);  sub_60 = log_2 = None
        exp_15 = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(mul_161, [1], True)
        mul_162 = torch.ops.aten.mul.Tensor(exp_15, sum_34);  exp_15 = sum_34 = None
        sub_106 = torch.ops.aten.sub.Tensor(mul_161, mul_162);  mul_161 = mul_162 = None
        convert_element_type_55 = torch.ops.prims.convert_element_type.default(sub_106, torch.bfloat16);  sub_106 = None
        mm_15 = torch.ops.aten.mm.default(convert_element_type_55, permute_10);  convert_element_type_55 = None
        scatter_7 = torch.ops.aten.scatter.value(full_1, 1, where_32, -1.0);  where_32 = None
        where_33 = torch.ops.aten.where.self(ne_71, div_1, full_default_3);  ne_71 = None
        mul_163 = torch.ops.aten.mul.Tensor(scatter_7, where_33);  scatter_7 = where_33 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        sub_54 = torch.ops.aten.sub.Tensor(convert_element_type_6, amax_1);  convert_element_type_6 = amax_1 = None
        sub_55 = torch.ops.aten.sub.Tensor(sub_54, log_1);  sub_54 = log_1 = None
        exp_16 = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_35 = torch.ops.aten.sum.dim_IntList(mul_163, [1], True)
        mul_164 = torch.ops.aten.mul.Tensor(exp_16, sum_35);  exp_16 = sum_35 = None
        sub_107 = torch.ops.aten.sub.Tensor(mul_163, mul_164);  mul_163 = mul_164 = None
        convert_element_type_58 = torch.ops.prims.convert_element_type.default(sub_107, torch.bfloat16);  sub_107 = None
        mm_16 = torch.ops.aten.mm.default(convert_element_type_58, permute_10);  convert_element_type_58 = None
        scatter_8 = torch.ops.aten.scatter.value(full_1, 1, where_34, -1.0);  full_1 = where_34 = None
        where_35 = torch.ops.aten.where.self(ne_73, div_1, full_default_3);  ne_73 = div_1 = full_default_3 = None
        mul_165 = torch.ops.aten.mul.Tensor(scatter_8, where_35);  scatter_8 = where_35 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        sub_48 = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        sub_49 = torch.ops.aten.sub.Tensor(sub_48, log);  sub_48 = log = None
        exp_17 = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_36 = torch.ops.aten.sum.dim_IntList(mul_165, [1], True)
        mul_166 = torch.ops.aten.mul.Tensor(exp_17, sum_36);  exp_17 = sum_36 = None
        sub_108 = torch.ops.aten.sub.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
        convert_element_type_61 = torch.ops.prims.convert_element_type.default(sub_108, torch.bfloat16);  sub_108 = None
        mm_17 = torch.ops.aten.mm.default(convert_element_type_61, permute_10);  convert_element_type_61 = permute_10 = None
        cat = torch.ops.aten.cat.default([mm_17, mm_16, mm_15, mm_14, mm_13, mm_12, mm_11, mm_10, mm_9]);  mm_17 = mm_16 = mm_15 = mm_14 = mm_13 = mm_12 = mm_11 = mm_10 = mm_9 = None
        view_11 = torch.ops.aten.view.default(cat, [primals_7, primals_8, 2560]);  cat = primals_7 = primals_8 = None
        return (None, None, None, None, None, None, None, None, view_11, None)
        
def load_args(reader):
    reader.symint(17)  # primals_7
    reader.symint(149)  # primals_8
    reader.symint(17)  # primals_2
    reader.symint(149)  # primals_3
    reader.symint(282)  # floordiv
    reader.symint(277)  # sym_size_int_9
    buf0 = reader.storage(None, 524416*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (((s5*s6 + 8)//9), 262208), dtype=torch.bfloat16, is_leaf=True)  # mm
    buf1 = reader.storage(None, 4*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf1, (((s5*s6 + 8)//9), 1), is_leaf=True)  # amax
    buf2 = reader.storage(None, 4*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf2, (((s5*s6 + 8)//9), 1), is_leaf=True)  # log
    buf3 = reader.storage(None, 524416*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (((s5*s6 + 8)//9), 262208), dtype=torch.bfloat16, is_leaf=True)  # mm_1
    buf4 = reader.storage(None, 4*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf4, (((s5*s6 + 8)//9), 1), is_leaf=True)  # amax_1
    buf5 = reader.storage(None, 4*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf5, (((s5*s6 + 8)//9), 1), is_leaf=True)  # log_1
    buf6 = reader.storage(None, 524416*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (((s5*s6 + 8)//9), 262208), dtype=torch.bfloat16, is_leaf=True)  # mm_2
    buf7 = reader.storage(None, 4*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf7, (((s5*s6 + 8)//9), 1), is_leaf=True)  # amax_2
    buf8 = reader.storage(None, 4*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf8, (((s5*s6 + 8)//9), 1), is_leaf=True)  # log_2
    buf9 = reader.storage(None, 524416*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf9, (((s5*s6 + 8)//9), 262208), dtype=torch.bfloat16, is_leaf=True)  # mm_3
    buf10 = reader.storage(None, 4*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf10, (((s5*s6 + 8)//9), 1), is_leaf=True)  # amax_3
    buf11 = reader.storage(None, 4*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf11, (((s5*s6 + 8)//9), 1), is_leaf=True)  # log_3
    buf12 = reader.storage(None, 524416*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf12, (((s5*s6 + 8)//9), 262208), dtype=torch.bfloat16, is_leaf=True)  # mm_4
    buf13 = reader.storage(None, 4*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf13, (((s5*s6 + 8)//9), 1), is_leaf=True)  # amax_4
    buf14 = reader.storage(None, 4*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf14, (((s5*s6 + 8)//9), 1), is_leaf=True)  # log_4
    buf15 = reader.storage(None, 524416*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf15, (((s5*s6 + 8)//9), 262208), dtype=torch.bfloat16, is_leaf=True)  # mm_5
    buf16 = reader.storage(None, 4*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf16, (((s5*s6 + 8)//9), 1), is_leaf=True)  # amax_5
    buf17 = reader.storage(None, 4*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf17, (((s5*s6 + 8)//9), 1), is_leaf=True)  # log_5
    buf18 = reader.storage(None, 524416*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf18, (((s5*s6 + 8)//9), 262208), dtype=torch.bfloat16, is_leaf=True)  # mm_6
    buf19 = reader.storage(None, 4*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf19, (((s5*s6 + 8)//9), 1), is_leaf=True)  # amax_6
    buf20 = reader.storage(None, 4*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf20, (((s5*s6 + 8)//9), 1), is_leaf=True)  # log_6
    buf21 = reader.storage(None, 524416*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf21, (((s5*s6 + 8)//9), 262208), dtype=torch.bfloat16, is_leaf=True)  # mm_7
    buf22 = reader.storage(None, 4*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf22, (((s5*s6 + 8)//9), 1), is_leaf=True)  # amax_7
    buf23 = reader.storage(None, 4*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf23, (((s5*s6 + 8)//9), 1), is_leaf=True)  # log_7
    buf24 = reader.storage(None, 524416*s5*s6 - 4195328*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf24, (s5*s6 - 8*(((s5*s6 + 8)//9)), 262208), dtype=torch.bfloat16, is_leaf=True)  # mm_8
    buf25 = reader.storage(None, 4*s5*s6 - 32*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf25, (s5*s6 - 8*(((s5*s6 + 8)//9)), 1), is_leaf=True)  # amax_8
    buf26 = reader.storage(None, 4*s5*s6 - 32*(((s5*s6 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf26, (s5*s6 - 8*(((s5*s6 + 8)//9)), 1), is_leaf=True)  # log_8
    buf27 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf27, (), is_leaf=True)  # convert_element_type_36
    buf28 = reader.storage(None, 17*s1 - 8*(((17*s1 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf28, (17*s1 - 8*(((17*s1 + 8)//9)), 1), dtype=torch.bool, is_leaf=True)  # ne_57
    buf29 = reader.storage(None, 136*s1 - 64*(((17*s1 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf29, (17*s1 - 8*(((17*s1 + 8)//9)), 1), dtype=torch.int64, is_leaf=True)  # where_18
    buf30 = reader.storage(None, 1342504960, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf30, (262208, 2560), dtype=torch.bfloat16, is_leaf=True)  # permute_10
    buf31 = reader.storage(None, ((17*s1 + 8)//9), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf31, (((17*s1 + 8)//9), 1), dtype=torch.bool, is_leaf=True)  # ne_59
    buf32 = reader.storage(None, 8*(((17*s1 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf32, (((17*s1 + 8)//9), 1), dtype=torch.int64, is_leaf=True)  # where_20
    buf33 = reader.storage(None, ((17*s1 + 8)//9), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf33, (((17*s1 + 8)//9), 1), dtype=torch.bool, is_leaf=True)  # ne_61
    buf34 = reader.storage(None, 8*(((17*s1 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf34, (((17*s1 + 8)//9), 1), dtype=torch.int64, is_leaf=True)  # where_22
    buf35 = reader.storage(None, ((17*s1 + 8)//9), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf35, (((17*s1 + 8)//9), 1), dtype=torch.bool, is_leaf=True)  # ne_63
    buf36 = reader.storage(None, 8*(((17*s1 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf36, (((17*s1 + 8)//9), 1), dtype=torch.int64, is_leaf=True)  # where_24
    buf37 = reader.storage(None, ((17*s1 + 8)//9), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf37, (((17*s1 + 8)//9), 1), dtype=torch.bool, is_leaf=True)  # ne_65
    buf38 = reader.storage(None, 8*(((17*s1 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf38, (((17*s1 + 8)//9), 1), dtype=torch.int64, is_leaf=True)  # where_26
    buf39 = reader.storage(None, ((17*s1 + 8)//9), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf39, (((17*s1 + 8)//9), 1), dtype=torch.bool, is_leaf=True)  # ne_67
    buf40 = reader.storage(None, 8*(((17*s1 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf40, (((17*s1 + 8)//9), 1), dtype=torch.int64, is_leaf=True)  # where_28
    buf41 = reader.storage(None, ((17*s1 + 8)//9), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf41, (((17*s1 + 8)//9), 1), dtype=torch.bool, is_leaf=True)  # ne_69
    buf42 = reader.storage(None, 8*(((17*s1 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf42, (((17*s1 + 8)//9), 1), dtype=torch.int64, is_leaf=True)  # where_30
    buf43 = reader.storage(None, ((17*s1 + 8)//9), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf43, (((17*s1 + 8)//9), 1), dtype=torch.bool, is_leaf=True)  # ne_71
    buf44 = reader.storage(None, 8*(((17*s1 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf44, (((17*s1 + 8)//9), 1), dtype=torch.int64, is_leaf=True)  # where_32
    buf45 = reader.storage(None, ((17*s1 + 8)//9), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf45, (((17*s1 + 8)//9), 1), dtype=torch.bool, is_leaf=True)  # ne_73
    buf46 = reader.storage(None, 8*(((17*s1 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf46, (((17*s1 + 8)//9), 1), dtype=torch.int64, is_leaf=True)  # where_34
    buf47 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf47, (), is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='symbolic', check_str=None)
        # mod(*args)