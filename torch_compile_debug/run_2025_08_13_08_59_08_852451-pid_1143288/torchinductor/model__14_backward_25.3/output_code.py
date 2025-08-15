# AOT ID: ['14_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_root/k3/ck3sfc3dhc22svzn2x64pyhs2jozwqlivw4bzdtozl53tbangu6e.py
# Topologically Sorted Source Nodes: [scatter, cross_entropy_loss, float_9, cross_entropy_loss_8], Original ATen: [aten.div, aten.nll_loss_backward, aten.nll_loss_forward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_3
#   cross_entropy_loss_8 => sub_96, sub_97
#   float_9 => convert_element_type_34
#   scatter => scatter_upon_const_tensor
# Graph fragment:
#   %div_1 : [num_users=9] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_36), kwargs = {})
#   %scatter_upon_const_tensor : [num_users=1] = call_function[target=torch._inductor.fx_passes.post_grad.scatter_upon_const_tensor](args = (), kwargs = {shape: [%sym_size_int_9, 262208], background_val: 0, dtype: torch.float32, dim: 1, selector: %where_18, val: -1.0})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_19 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_57, %div_1, %full_default_3), kwargs = {})
#   %mul_149 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_upon_const_tensor, %where_19), kwargs = {})
#   %convert_element_type_34 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_8, torch.float32), kwargs = {})
#   %sub_96 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_34, %amax_8), kwargs = {})
#   %sub_97 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_96, %log_8), kwargs = {})
#   %exp_9 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_97,), kwargs = {})
#   %sum_28 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_149, [1], True), kwargs = {})
#   %mul_150 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_9, %sum_28), kwargs = {})
#   %sub_100 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_149, %mul_150), kwargs = {})
#   %convert_element_type_37 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_100, torch.bfloat16), kwargs = {})
triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_0 = async_compile.triton('triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 524288},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 1, 'backend_hash': '23E61D9BD7D1F06E5EE613BAAD805B205CC78A2D41FEC496A5ED9AB34FB056A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 262208
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp7 = tl.load(in_ptr2 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp9 = tl.load(in_ptr3 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = r0_1
        tmp2 = tmp0 == tmp1
        tmp3 = -1.0
        tmp4 = 0.0
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp11 = (tmp8 / tmp10)
        tmp12 = tl.where(tmp6, tmp11, tmp4)
        tmp13 = tmp5 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tmp22 = tl.load(in_ptr2 + (0))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
    tmp24 = tl.load(in_ptr3 + (0))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp31 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp29 = tl.load(in_out_ptr0 + (r0_1 + 262208*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = r0_1
        tmp18 = tmp0 == tmp17
        tmp19 = -1.0
        tmp20 = 0.0
        tmp21 = tl.where(tmp18, tmp19, tmp20)
        tmp26 = (tmp23 / tmp25)
        tmp27 = tl.where(tmp6, tmp26, tmp20)
        tmp28 = tmp21 * tmp27
        tmp30 = tmp29.to(tl.float32)
        tmp32 = tmp30 - tmp31
        tmp34 = tmp32 - tmp33
        tmp35 = tl_math.exp(tmp34)
        tmp36 = tmp35 * tmp15
        tmp37 = tmp28 - tmp36
        tmp38 = tmp37.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_1 + 262208*x0), tmp38, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/fk/cfkwxp2ksiqarmfhf2qpvfolj2pntxhjhw6gba3zp4fcdolbxctx.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_1 : [num_users=8] = call_function[target=torch.ops.aten.full.default](args = ([%floordiv, 262208], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter_1 : [num_users=1] = call_function[target=torch.ops.aten.scatter.value](args = (%full_1, 1, %where_20, -1.0), kwargs = {})
triton_poi_fused_nll_loss_backward_1 = async_compile.triton('triton_poi_fused_nll_loss_backward_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '23E61D9BD7D1F06E5EE613BAAD805B205CC78A2D41FEC496A5ED9AB34FB056A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_nll_loss_backward_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/m7/cm7zeelg4njhhoxhb5cvppkmd2apmrlpf3hdj7gwo6az7pjtxkae.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_1 : [num_users=8] = call_function[target=torch.ops.aten.full.default](args = ([%floordiv, 262208], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter_1 : [num_users=1] = call_function[target=torch.ops.aten.scatter.value](args = (%full_1, 1, %where_20, -1.0), kwargs = {})
triton_poi_fused_nll_loss_backward_2 = async_compile.triton('triton_poi_fused_nll_loss_backward_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_2', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '23E61D9BD7D1F06E5EE613BAAD805B205CC78A2D41FEC496A5ED9AB34FB056A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_nll_loss_backward_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.device_assert(((0 <= tmp0) & (tmp0 < 262208)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 262208")
    tmp2 = -1.0
    tl.store(out_ptr0 + (tmp0 + 262208*x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/xa/cxai2yzo5w5l5oydv26yhidy7adaj7bbqgsz4bjquxtfpoj6y7md.py
# Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_3
# Graph fragment:
#   %div_1 : [num_users=9] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_36), kwargs = {})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_21 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_59, %div_1, %full_default_3), kwargs = {})
#   %mul_151 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_1, %where_21), kwargs = {})
#   %sum_29 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_151, [1], True), kwargs = {})
triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3 = async_compile.triton('triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 524288},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '23E61D9BD7D1F06E5EE613BAAD805B205CC78A2D41FEC496A5ED9AB34FB056A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 262208
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_ptr2 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp4 = tl.load(in_ptr3 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 262208*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = (tmp3 / tmp5)
        tmp7 = 0.0
        tmp8 = tl.where(tmp1, tmp6, tmp7)
        tmp9 = tmp0 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/oh/cohokcriviliqyzcfbs22oayqmd3nycb6ftmi52wvg76lth43qm4.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, float_8, cross_entropy_loss_7], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_3
#   cross_entropy_loss_7 => sub_90, sub_91
#   float_8 => convert_element_type_30
# Graph fragment:
#   %div_1 : [num_users=9] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_36), kwargs = {})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_21 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_59, %div_1, %full_default_3), kwargs = {})
#   %mul_151 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_1, %where_21), kwargs = {})
#   %convert_element_type_30 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_7, torch.float32), kwargs = {})
#   %sub_90 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_30, %amax_7), kwargs = {})
#   %sub_91 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_90, %log_7), kwargs = {})
#   %exp_10 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_91,), kwargs = {})
#   %mul_152 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_10, %sum_29), kwargs = {})
#   %sub_101 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_151, %mul_152), kwargs = {})
#   %convert_element_type_40 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_101, torch.bfloat16), kwargs = {})
triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4 = async_compile.triton('triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '23E61D9BD7D1F06E5EE613BAAD805B205CC78A2D41FEC496A5ED9AB34FB056A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 262208
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_ptr2 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tl.load(in_ptr3 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp10 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = (tmp3 / tmp5)
    tmp7 = 0.0
    tmp8 = tl.where(tmp1, tmp6, tmp7)
    tmp9 = tmp0 * tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 - tmp12
    tmp15 = tmp13 - tmp14
    tmp16 = tl_math.exp(tmp15)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp20, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_7, primals_8, primals_2, primals_3, floordiv, sym_size_int_9, mm, amax, log, mm_1, amax_1, log_1, mm_2, amax_2, log_2, mm_3, amax_3, log_3, mm_4, amax_4, log_4, mm_5, amax_5, log_5, mm_6, amax_6, log_6, mm_7, amax_7, log_7, mm_8, amax_8, log_8, convert_element_type_36, ne_57, where_18, permute_10, ne_59, where_20, ne_61, where_22, ne_63, where_24, ne_65, where_26, ne_67, where_28, ne_69, where_30, ne_71, where_32, ne_73, where_34, tangents_1 = args
    args.clear()
    s5 = primals_7
    s6 = primals_8
    s1 = primals_3
    assert_size_stride(mm, ((8 + s5*s6) // 9, 262208), (262208, 1))
    assert_size_stride(amax, ((8 + s5*s6) // 9, 1), (1, 1))
    assert_size_stride(log, ((8 + s5*s6) // 9, 1), (1, 1))
    assert_size_stride(mm_1, ((8 + s5*s6) // 9, 262208), (262208, 1))
    assert_size_stride(amax_1, ((8 + s5*s6) // 9, 1), (1, 1))
    assert_size_stride(log_1, ((8 + s5*s6) // 9, 1), (1, 1))
    assert_size_stride(mm_2, ((8 + s5*s6) // 9, 262208), (262208, 1))
    assert_size_stride(amax_2, ((8 + s5*s6) // 9, 1), (1, 1))
    assert_size_stride(log_2, ((8 + s5*s6) // 9, 1), (1, 1))
    assert_size_stride(mm_3, ((8 + s5*s6) // 9, 262208), (262208, 1))
    assert_size_stride(amax_3, ((8 + s5*s6) // 9, 1), (1, 1))
    assert_size_stride(log_3, ((8 + s5*s6) // 9, 1), (1, 1))
    assert_size_stride(mm_4, ((8 + s5*s6) // 9, 262208), (262208, 1))
    assert_size_stride(amax_4, ((8 + s5*s6) // 9, 1), (1, 1))
    assert_size_stride(log_4, ((8 + s5*s6) // 9, 1), (1, 1))
    assert_size_stride(mm_5, ((8 + s5*s6) // 9, 262208), (262208, 1))
    assert_size_stride(amax_5, ((8 + s5*s6) // 9, 1), (1, 1))
    assert_size_stride(log_5, ((8 + s5*s6) // 9, 1), (1, 1))
    assert_size_stride(mm_6, ((8 + s5*s6) // 9, 262208), (262208, 1))
    assert_size_stride(amax_6, ((8 + s5*s6) // 9, 1), (1, 1))
    assert_size_stride(log_6, ((8 + s5*s6) // 9, 1), (1, 1))
    assert_size_stride(mm_7, ((8 + s5*s6) // 9, 262208), (262208, 1))
    assert_size_stride(amax_7, ((8 + s5*s6) // 9, 1), (1, 1))
    assert_size_stride(log_7, ((8 + s5*s6) // 9, 1), (1, 1))
    assert_size_stride(mm_8, ((-8)*((8 + s5*s6) // 9) + s5*s6, 262208), (262208, 1))
    assert_size_stride(amax_8, ((-8)*((8 + s5*s6) // 9) + s5*s6, 1), (1, 1))
    assert_size_stride(log_8, ((-8)*((8 + s5*s6) // 9) + s5*s6, 1), (1, 1))
    assert_size_stride(convert_element_type_36, (), ())
    assert_size_stride(ne_57, ((-8)*((8 + 17*s1) // 9) + 17*s1, 1), (1, 1))
    assert_size_stride(where_18, ((-8)*((8 + 17*s1) // 9) + 17*s1, 1), (1, 1))
    assert_size_stride(permute_10, (262208, 2560), (2560, 1))
    assert_size_stride(ne_59, ((8 + 17*s1) // 9, 1), (1, 1))
    assert_size_stride(where_20, ((8 + 17*s1) // 9, 1), (1, 1))
    assert_size_stride(ne_61, ((8 + 17*s1) // 9, 1), (1, 1))
    assert_size_stride(where_22, ((8 + 17*s1) // 9, 1), (1, 1))
    assert_size_stride(ne_63, ((8 + 17*s1) // 9, 1), (1, 1))
    assert_size_stride(where_24, ((8 + 17*s1) // 9, 1), (1, 1))
    assert_size_stride(ne_65, ((8 + 17*s1) // 9, 1), (1, 1))
    assert_size_stride(where_26, ((8 + 17*s1) // 9, 1), (1, 1))
    assert_size_stride(ne_67, ((8 + 17*s1) // 9, 1), (1, 1))
    assert_size_stride(where_28, ((8 + 17*s1) // 9, 1), (1, 1))
    assert_size_stride(ne_69, ((8 + 17*s1) // 9, 1), (1, 1))
    assert_size_stride(where_30, ((8 + 17*s1) // 9, 1), (1, 1))
    assert_size_stride(ne_71, ((8 + 17*s1) // 9, 1), (1, 1))
    assert_size_stride(where_32, ((8 + 17*s1) // 9, 1), (1, 1))
    assert_size_stride(ne_73, ((8 + 17*s1) // 9, 1), (1, 1))
    assert_size_stride(where_34, ((8 + 17*s1) // 9, 1), (1, 1))
    assert_size_stride(tangents_1, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = mm_8; del mm_8  # reuse
        # Topologically Sorted Source Nodes: [scatter, cross_entropy_loss, float_9, cross_entropy_loss_8], Original ATen: [aten.div, aten.nll_loss_backward, aten.nll_loss_forward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_0_xnumel = (-8)*((8 + s5*s6) // 9) + s5*s6
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_0.run(buf1, where_18, ne_57, tangents_1, convert_element_type_36, amax_8, log_8, triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_0_xnumel, 262208, stream=stream0)
        del amax_8
        del log_8
        del ne_57
        del where_18
        buf43 = empty_strided_cuda((s5*s6, 2560), (2560, 1), torch.bfloat16)
        buf2 = reinterpret_tensor(buf43, ((-8)*((8 + s5*s6) // 9) + s5*s6, 2560), (2560, 1), 20480*((8 + s5*s6) // 9))  # alias
        # Topologically Sorted Source Nodes: [scatter, cross_entropy_loss, float_9, cross_entropy_loss_8], Original ATen: [aten.div, aten.nll_loss_backward, aten.nll_loss_forward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data, aten.mm]
        extern_kernels.mm(buf1, permute_10, out=buf2)
        del buf1
        buf3 = empty_strided_cuda(((8 + s5*s6) // 9, 262208), (262208, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_1_xnumel = 262208*((8 + s5*s6) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_1.run(buf3, triton_poi_fused_nll_loss_backward_1_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = (8 + 17*s1) // 9
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(where_20, buf3, triton_poi_fused_nll_loss_backward_2_xnumel, stream=stream0)
        del where_20
        buf5 = empty_strided_cuda(((8 + s5*s6) // 9, 1), (1, (8 + s5*s6) // 9), torch.float32)
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel = (8 + s5*s6) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3.run(buf3, ne_59, tangents_1, convert_element_type_36, buf5, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel, 262208, stream=stream0)
        buf6 = mm_7; del mm_7  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_8, cross_entropy_loss_7], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4_xnumel = 262208*((8 + s5*s6) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4.run(buf6, buf3, ne_59, tangents_1, convert_element_type_36, amax_7, log_7, buf5, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4_xnumel, stream=stream0)
        del amax_7
        del log_7
        del ne_59
        buf7 = reinterpret_tensor(buf43, ((8 + s5*s6) // 9, 2560), (2560, 1), 17920*((8 + s5*s6) // 9))  # alias
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_8, cross_entropy_loss_7], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data, aten.mm]
        extern_kernels.mm(buf6, permute_10, out=buf7)
        del buf6
        buf8 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_1_xnumel = 262208*((8 + s5*s6) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_1.run(buf8, triton_poi_fused_nll_loss_backward_1_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = (8 + 17*s1) // 9
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(where_22, buf8, triton_poi_fused_nll_loss_backward_2_xnumel, stream=stream0)
        del where_22
        buf10 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel = (8 + s5*s6) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3.run(buf8, ne_61, tangents_1, convert_element_type_36, buf10, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel, 262208, stream=stream0)
        buf11 = mm_6; del mm_6  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_7, cross_entropy_loss_6], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4_xnumel = 262208*((8 + s5*s6) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4.run(buf11, buf8, ne_61, tangents_1, convert_element_type_36, amax_6, log_6, buf10, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4_xnumel, stream=stream0)
        del amax_6
        del log_6
        del ne_61
        buf12 = reinterpret_tensor(buf43, ((8 + s5*s6) // 9, 2560), (2560, 1), 15360*((8 + s5*s6) // 9))  # alias
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_7, cross_entropy_loss_6], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data, aten.mm]
        extern_kernels.mm(buf11, permute_10, out=buf12)
        del buf11
        buf13 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_1_xnumel = 262208*((8 + s5*s6) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_1.run(buf13, triton_poi_fused_nll_loss_backward_1_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = (8 + 17*s1) // 9
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(where_24, buf13, triton_poi_fused_nll_loss_backward_2_xnumel, stream=stream0)
        del where_24
        buf15 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel = (8 + s5*s6) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3.run(buf13, ne_63, tangents_1, convert_element_type_36, buf15, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel, 262208, stream=stream0)
        buf16 = mm_5; del mm_5  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_6, cross_entropy_loss_5], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4_xnumel = 262208*((8 + s5*s6) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4.run(buf16, buf13, ne_63, tangents_1, convert_element_type_36, amax_5, log_5, buf15, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4_xnumel, stream=stream0)
        del amax_5
        del log_5
        del ne_63
        buf17 = reinterpret_tensor(buf43, ((8 + s5*s6) // 9, 2560), (2560, 1), 12800*((8 + s5*s6) // 9))  # alias
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_6, cross_entropy_loss_5], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data, aten.mm]
        extern_kernels.mm(buf16, permute_10, out=buf17)
        del buf16
        buf18 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_1_xnumel = 262208*((8 + s5*s6) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_1.run(buf18, triton_poi_fused_nll_loss_backward_1_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = (8 + 17*s1) // 9
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(where_26, buf18, triton_poi_fused_nll_loss_backward_2_xnumel, stream=stream0)
        del where_26
        buf20 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel = (8 + s5*s6) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3.run(buf18, ne_65, tangents_1, convert_element_type_36, buf20, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel, 262208, stream=stream0)
        buf21 = mm_4; del mm_4  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_5, cross_entropy_loss_4], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4_xnumel = 262208*((8 + s5*s6) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4.run(buf21, buf18, ne_65, tangents_1, convert_element_type_36, amax_4, log_4, buf20, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4_xnumel, stream=stream0)
        del amax_4
        del log_4
        del ne_65
        buf22 = reinterpret_tensor(buf43, ((8 + s5*s6) // 9, 2560), (2560, 1), 10240*((8 + s5*s6) // 9))  # alias
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_5, cross_entropy_loss_4], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data, aten.mm]
        extern_kernels.mm(buf21, permute_10, out=buf22)
        del buf21
        buf23 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_1_xnumel = 262208*((8 + s5*s6) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_1.run(buf23, triton_poi_fused_nll_loss_backward_1_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = (8 + 17*s1) // 9
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(where_28, buf23, triton_poi_fused_nll_loss_backward_2_xnumel, stream=stream0)
        del where_28
        buf25 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel = (8 + s5*s6) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3.run(buf23, ne_67, tangents_1, convert_element_type_36, buf25, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel, 262208, stream=stream0)
        buf26 = mm_3; del mm_3  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_4, cross_entropy_loss_3], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4_xnumel = 262208*((8 + s5*s6) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4.run(buf26, buf23, ne_67, tangents_1, convert_element_type_36, amax_3, log_3, buf25, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4_xnumel, stream=stream0)
        del amax_3
        del log_3
        del ne_67
        buf27 = reinterpret_tensor(buf43, ((8 + s5*s6) // 9, 2560), (2560, 1), 7680*((8 + s5*s6) // 9))  # alias
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_4, cross_entropy_loss_3], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data, aten.mm]
        extern_kernels.mm(buf26, permute_10, out=buf27)
        del buf26
        buf28 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_1_xnumel = 262208*((8 + s5*s6) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_1.run(buf28, triton_poi_fused_nll_loss_backward_1_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = (8 + 17*s1) // 9
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(where_30, buf28, triton_poi_fused_nll_loss_backward_2_xnumel, stream=stream0)
        del where_30
        buf30 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel = (8 + s5*s6) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3.run(buf28, ne_69, tangents_1, convert_element_type_36, buf30, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel, 262208, stream=stream0)
        buf31 = mm_2; del mm_2  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_3, cross_entropy_loss_2], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4_xnumel = 262208*((8 + s5*s6) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4.run(buf31, buf28, ne_69, tangents_1, convert_element_type_36, amax_2, log_2, buf30, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4_xnumel, stream=stream0)
        del amax_2
        del log_2
        del ne_69
        buf32 = reinterpret_tensor(buf43, ((8 + s5*s6) // 9, 2560), (2560, 1), 5120*((8 + s5*s6) // 9))  # alias
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_3, cross_entropy_loss_2], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data, aten.mm]
        extern_kernels.mm(buf31, permute_10, out=buf32)
        del buf31
        buf33 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_1_xnumel = 262208*((8 + s5*s6) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_1.run(buf33, triton_poi_fused_nll_loss_backward_1_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = (8 + 17*s1) // 9
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(where_32, buf33, triton_poi_fused_nll_loss_backward_2_xnumel, stream=stream0)
        del where_32
        buf35 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel = (8 + s5*s6) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3.run(buf33, ne_71, tangents_1, convert_element_type_36, buf35, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel, 262208, stream=stream0)
        buf36 = mm_1; del mm_1  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_2, cross_entropy_loss_1], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4_xnumel = 262208*((8 + s5*s6) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4.run(buf36, buf33, ne_71, tangents_1, convert_element_type_36, amax_1, log_1, buf35, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4_xnumel, stream=stream0)
        del amax_1
        del log_1
        del ne_71
        buf37 = reinterpret_tensor(buf43, ((8 + s5*s6) // 9, 2560), (2560, 1), 2560*((8 + s5*s6) // 9))  # alias
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_2, cross_entropy_loss_1], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data, aten.mm]
        extern_kernels.mm(buf36, permute_10, out=buf37)
        del buf36
        buf38 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_1_xnumel = 262208*((8 + s5*s6) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_1.run(buf38, triton_poi_fused_nll_loss_backward_1_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = (8 + 17*s1) // 9
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(where_34, buf38, triton_poi_fused_nll_loss_backward_2_xnumel, stream=stream0)
        del where_34
        buf40 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel = (8 + s5*s6) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3.run(buf38, ne_73, tangents_1, convert_element_type_36, buf40, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3_xnumel, 262208, stream=stream0)
        buf41 = mm; del mm  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_1], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4_xnumel = 262208*((8 + s5*s6) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4.run(buf41, buf38, ne_73, tangents_1, convert_element_type_36, amax, log, buf40, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_nll_loss_backward_nll_loss_forward_4_xnumel, stream=stream0)
        del amax
        del buf38
        del buf40
        del convert_element_type_36
        del log
        del ne_73
        del tangents_1
        buf42 = reinterpret_tensor(buf43, ((8 + s5*s6) // 9, 2560), (2560, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [cross_entropy_loss, float_1], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data, aten.mm]
        extern_kernels.mm(buf41, permute_10, out=buf42)
        del buf41
        del permute_10
    return (None, None, None, None, None, None, None, None, reinterpret_tensor(buf43, (s5, s6, 2560), (2560*s6, 2560, 1), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_7 = 17
    primals_8 = 149
    primals_2 = 17
    primals_3 = 149
    floordiv = 282
    sym_size_int_9 = 277
    mm = rand_strided((282, 262208), (262208, 1), device='cuda:0', dtype=torch.bfloat16)
    amax = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_1 = rand_strided((282, 262208), (262208, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_1 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_1 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_2 = rand_strided((282, 262208), (262208, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_2 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_2 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_3 = rand_strided((282, 262208), (262208, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_3 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_3 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_4 = rand_strided((282, 262208), (262208, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_4 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_4 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_5 = rand_strided((282, 262208), (262208, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_5 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_5 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_6 = rand_strided((282, 262208), (262208, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_6 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_6 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_7 = rand_strided((282, 262208), (262208, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_7 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_7 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_8 = rand_strided((277, 262208), (262208, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_8 = rand_strided((277, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_8 = rand_strided((277, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_36 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    ne_57 = rand_strided((277, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_18 = rand_strided((277, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    permute_10 = rand_strided((262208, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    ne_59 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_20 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_61 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_22 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_63 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_24 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_65 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_26 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_67 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_28 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_69 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_30 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_71 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_32 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_73 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_34 = rand_strided((282, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_7, primals_8, primals_2, primals_3, floordiv, sym_size_int_9, mm, amax, log, mm_1, amax_1, log_1, mm_2, amax_2, log_2, mm_3, amax_3, log_3, mm_4, amax_4, log_4, mm_5, amax_5, log_5, mm_6, amax_6, log_6, mm_7, amax_7, log_7, mm_8, amax_8, log_8, convert_element_type_36, ne_57, where_18, permute_10, ne_59, where_20, ne_61, where_22, ne_63, where_24, ne_65, where_26, ne_67, where_28, ne_69, where_30, ne_71, where_32, ne_73, where_34, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
