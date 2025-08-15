# AOT ID: ['14_forward']
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


# kernel path: /tmp/torchinductor_root/wg/cwg7oyzownpdawmd2elmigfi65dytmsbsvhltf5wkycnhnsqwxou.py
# Topologically Sorted Source Nodes: [_generalized_scatter], Original ATen: []
# Source node to ATen node mapping:
#   _generalized_scatter => slice_scatter_default
# Graph fragment:
#   %slice_scatter_default : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%permute, %slice_1, 1, 0, -1), kwargs = {})
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'out_ptr0': '*i64', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '23E61D9BD7D1F06E5EE613BAAD805B205CC78A2D41FEC496A5ED9AB34FB056A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x2 = xindex
    tmp4 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = (-1) + ks0
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (1 + x2), xmask & tmp2, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x2), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/5o/c5o33c5deewv3z7e4jxbww7f7x6eu7f5ogfgtxp67jlt2jpoeibz.py
# Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_1 => full_default, index_put
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], -100), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%slice_6, [%eq_26], %full_default), kwargs = {})
triton_poi_fused_index_put_lift_fresh_1 = async_compile.triton('triton_poi_fused_index_put_lift_fresh_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'out_ptr1': '*i64', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_1', 'mutated_arg_names': ['out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '23E61D9BD7D1F06E5EE613BAAD805B205CC78A2D41FEC496A5ED9AB34FB056A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_put_lift_fresh_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = ks0
    tmp5 = tmp3 < tmp4
    tmp6 = tl.load(in_ptr1 + (1 + x0 + ks1*x1), xmask & tmp5, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tl.full([1], -100, tl.int64)
    tmp10 = tl.where(tmp2, tmp9, tmp8)
    tl.store(out_ptr1 + (x0 + ks1*x1), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/a7/ca7n42dknjzuxcqgoaso6j2c6lewf3chwhinrzdhopbfi6k3txhe.py
# Topologically Sorted Source Nodes: [float_1, , cross_entropy_loss], Original ATen: [aten._to_copy, prims.prepare_softmax_online, aten._log_softmax]
# Source node to ATen node mapping:
#    => prepare_softmax_online_default_8
#   cross_entropy_loss => log
#   float_1 => convert_element_type_2
# Graph fragment:
#   %convert_element_type_2 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %prepare_softmax_online_default_8 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %log : [num_users=2] = call_function[target=torch.ops.aten.log.default](args = (%getitem_116,), kwargs = {})
triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2 = async_compile.triton('triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '23E61D9BD7D1F06E5EE613BAAD805B205CC78A2D41FEC496A5ED9AB34FB056A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 262208
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp3_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 262208*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])

        _tmp3_max_next, _tmp3_sum_next = triton_helpers.online_softmax_combine(
            _tmp3_max, _tmp3_sum, tmp2, False
        )

        _tmp3_max = tl.where(r0_mask & xmask, _tmp3_max_next, _tmp3_max)
        _tmp3_sum = tl.where(r0_mask & xmask, _tmp3_sum_next, _tmp3_sum)

    tmp5, tmp6 = triton_helpers.online_softmax_reduce(
        _tmp3_max, _tmp3_sum, 1, False)
    tmp5 = tmp5[:, None]
    tmp6 = tmp6[:, None]
    tmp3 = tmp5
    tmp4 = tmp6
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tmp7 = tl_math.log(tmp4)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ns/cnsdf5y6gijy5fn6xoilzfdkmb6thy3f5cucyhro2welkxpnqtck.py
# Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2, full_default_3, ne_6, neg, sum_3, where_1
# Graph fragment:
#   %ne_6 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_18, -100), kwargs = {})
#   %full_default_2 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_6, %neg, %full_default_3), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %ne_73 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_17, -100), kwargs = {})
#   %where_34 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_73, %unsqueeze_17, %full_default_2), kwargs = {})
triton_red_fused_nll_loss_backward_nll_loss_forward_3 = async_compile.triton('triton_red_fused_nll_loss_backward_nll_loss_forward_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_backward_nll_loss_forward_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '23E61D9BD7D1F06E5EE613BAAD805B205CC78A2D41FEC496A5ED9AB34FB056A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_backward_nll_loss_forward_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp5 = tl.load(in_ptr0 + ((r0_0 % (17*ks0))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp0 = (r0_0 % ks0)
        tmp1 = ks1
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to((r0_0 % (17*ks0)), [XBLOCK, R0_BLOCK])), r0_mask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, R0_BLOCK], 262208, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 262208)) | ~(r0_mask), "index out of bounds: 0 <= tmp15 < 262208")
        tmp17 = tl.load(in_ptr1 + (tmp15 + 262208*r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 - tmp19
        tmp22 = tmp20 - tmp21
        tmp23 = -tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp9, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask, tmp28, _tmp27)
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp9, r0_mask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, r0_mask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/3u/c3uqv4j4a2sn62yicmf2fbkyjvwhae4zewelotuic3lp5q4ytk3w.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_1], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2, full_default_3
#   cross_entropy_loss_1 => ne_12, neg_1, sum_6, where_3
# Graph fragment:
#   %full_default_2 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_12 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_28, -100), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_1,), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_12, %neg_1, %full_default_3), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_3,), kwargs = {})
#   %ne_71 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_16, -100), kwargs = {})
#   %where_32 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_71, %unsqueeze_16, %full_default_2), kwargs = {})
triton_red_fused_nll_loss_backward_nll_loss_forward_4 = async_compile.triton('triton_red_fused_nll_loss_backward_nll_loss_forward_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_backward_nll_loss_forward_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '23E61D9BD7D1F06E5EE613BAAD805B205CC78A2D41FEC496A5ED9AB34FB056A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_backward_nll_loss_forward_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp5 = tl.load(in_ptr0 + (((r0_0 + ((8 + 17*ks0) // 9)) % (17*ks0))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp0 = ((r0_0 + ((8 + 17*ks0) // 9)) % ks0)
        tmp1 = ks1
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(((r0_0 + ((8 + 17*ks0) // 9)) % (17*ks0)), [XBLOCK, R0_BLOCK])), r0_mask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, R0_BLOCK], 262208, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 262208)) | ~(r0_mask), "index out of bounds: 0 <= tmp15 < 262208")
        tmp17 = tl.load(in_ptr1 + (tmp15 + 262208*r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 - tmp19
        tmp22 = tmp20 - tmp21
        tmp23 = -tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp9, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask, tmp28, _tmp27)
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp9, r0_mask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, r0_mask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/lq/clqypeit5qyexe5kusptbew3ly6zkdbr62nhu65j2rao27rvrlxl.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_2], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2, full_default_3
#   cross_entropy_loss_2 => ne_18, neg_2, sum_9, where_5
# Graph fragment:
#   %full_default_2 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_18 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_38, -100), kwargs = {})
#   %neg_2 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_2,), kwargs = {})
#   %where_5 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_18, %neg_2, %full_default_3), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_5,), kwargs = {})
#   %ne_69 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_15, -100), kwargs = {})
#   %where_30 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_69, %unsqueeze_15, %full_default_2), kwargs = {})
triton_red_fused_nll_loss_backward_nll_loss_forward_5 = async_compile.triton('triton_red_fused_nll_loss_backward_nll_loss_forward_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_backward_nll_loss_forward_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '23E61D9BD7D1F06E5EE613BAAD805B205CC78A2D41FEC496A5ED9AB34FB056A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_backward_nll_loss_forward_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp5 = tl.load(in_ptr0 + (((r0_0 + 2*((8 + 17*ks0) // 9)) % (17*ks0))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp0 = ((r0_0 + 2*((8 + 17*ks0) // 9)) % ks0)
        tmp1 = ks1
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(((r0_0 + 2*((8 + 17*ks0) // 9)) % (17*ks0)), [XBLOCK, R0_BLOCK])), r0_mask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, R0_BLOCK], 262208, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 262208)) | ~(r0_mask), "index out of bounds: 0 <= tmp15 < 262208")
        tmp17 = tl.load(in_ptr1 + (tmp15 + 262208*r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 - tmp19
        tmp22 = tmp20 - tmp21
        tmp23 = -tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp9, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask, tmp28, _tmp27)
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp9, r0_mask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, r0_mask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/kd/ckd5xynkogg5w6m35ier6tfms3kvrodf5dqy2g7uzqb77igy66r2.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_3], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2, full_default_3
#   cross_entropy_loss_3 => ne_24, neg_3, sum_12, where_7
# Graph fragment:
#   %full_default_2 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_24 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_48, -100), kwargs = {})
#   %neg_3 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_3,), kwargs = {})
#   %where_7 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_24, %neg_3, %full_default_3), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_7,), kwargs = {})
#   %ne_67 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_14, -100), kwargs = {})
#   %where_28 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_67, %unsqueeze_14, %full_default_2), kwargs = {})
triton_red_fused_nll_loss_backward_nll_loss_forward_6 = async_compile.triton('triton_red_fused_nll_loss_backward_nll_loss_forward_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_backward_nll_loss_forward_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '23E61D9BD7D1F06E5EE613BAAD805B205CC78A2D41FEC496A5ED9AB34FB056A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_backward_nll_loss_forward_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp5 = tl.load(in_ptr0 + (((r0_0 + 3*((8 + 17*ks0) // 9)) % (17*ks0))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp0 = ((r0_0 + 3*((8 + 17*ks0) // 9)) % ks0)
        tmp1 = ks1
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(((r0_0 + 3*((8 + 17*ks0) // 9)) % (17*ks0)), [XBLOCK, R0_BLOCK])), r0_mask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, R0_BLOCK], 262208, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 262208)) | ~(r0_mask), "index out of bounds: 0 <= tmp15 < 262208")
        tmp17 = tl.load(in_ptr1 + (tmp15 + 262208*r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 - tmp19
        tmp22 = tmp20 - tmp21
        tmp23 = -tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp9, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask, tmp28, _tmp27)
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp9, r0_mask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, r0_mask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/oa/coa45ue7rxx274ahinrp4cpozyo65xatajkvzkiieqemnuc7b52f.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_4], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2, full_default_3
#   cross_entropy_loss_4 => ne_30, neg_4, sum_15, where_9
# Graph fragment:
#   %full_default_2 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_30 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_58, -100), kwargs = {})
#   %neg_4 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_4,), kwargs = {})
#   %where_9 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_30, %neg_4, %full_default_3), kwargs = {})
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_9,), kwargs = {})
#   %ne_65 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_13, -100), kwargs = {})
#   %where_26 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_65, %unsqueeze_13, %full_default_2), kwargs = {})
triton_red_fused_nll_loss_backward_nll_loss_forward_7 = async_compile.triton('triton_red_fused_nll_loss_backward_nll_loss_forward_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_backward_nll_loss_forward_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '23E61D9BD7D1F06E5EE613BAAD805B205CC78A2D41FEC496A5ED9AB34FB056A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_backward_nll_loss_forward_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp5 = tl.load(in_ptr0 + (((r0_0 + 4*((8 + 17*ks0) // 9)) % (17*ks0))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp0 = ((r0_0 + 4*((8 + 17*ks0) // 9)) % ks0)
        tmp1 = ks1
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(((r0_0 + 4*((8 + 17*ks0) // 9)) % (17*ks0)), [XBLOCK, R0_BLOCK])), r0_mask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, R0_BLOCK], 262208, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 262208)) | ~(r0_mask), "index out of bounds: 0 <= tmp15 < 262208")
        tmp17 = tl.load(in_ptr1 + (tmp15 + 262208*r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 - tmp19
        tmp22 = tmp20 - tmp21
        tmp23 = -tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp9, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask, tmp28, _tmp27)
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp9, r0_mask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, r0_mask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/k5/ck5azxad2om5kfegulzqfzctp2iwptxvbnvcqwb2k4xp42jiukcn.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_5], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2, full_default_3
#   cross_entropy_loss_5 => ne_36, neg_5, sum_18, where_11
# Graph fragment:
#   %full_default_2 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_36 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_68, -100), kwargs = {})
#   %neg_5 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_5,), kwargs = {})
#   %where_11 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_36, %neg_5, %full_default_3), kwargs = {})
#   %sum_18 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_11,), kwargs = {})
#   %ne_63 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_12, -100), kwargs = {})
#   %where_24 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_63, %unsqueeze_12, %full_default_2), kwargs = {})
triton_red_fused_nll_loss_backward_nll_loss_forward_8 = async_compile.triton('triton_red_fused_nll_loss_backward_nll_loss_forward_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_backward_nll_loss_forward_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '23E61D9BD7D1F06E5EE613BAAD805B205CC78A2D41FEC496A5ED9AB34FB056A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_backward_nll_loss_forward_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp5 = tl.load(in_ptr0 + (((r0_0 + 5*((8 + 17*ks0) // 9)) % (17*ks0))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp0 = ((r0_0 + 5*((8 + 17*ks0) // 9)) % ks0)
        tmp1 = ks1
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(((r0_0 + 5*((8 + 17*ks0) // 9)) % (17*ks0)), [XBLOCK, R0_BLOCK])), r0_mask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, R0_BLOCK], 262208, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 262208)) | ~(r0_mask), "index out of bounds: 0 <= tmp15 < 262208")
        tmp17 = tl.load(in_ptr1 + (tmp15 + 262208*r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 - tmp19
        tmp22 = tmp20 - tmp21
        tmp23 = -tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp9, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask, tmp28, _tmp27)
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp9, r0_mask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, r0_mask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/d2/cd2dimqzgvdyy3sxpzoepidbqton44ji4ry6bplzfyoyrudimwaa.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_6], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2, full_default_3
#   cross_entropy_loss_6 => ne_42, neg_6, sum_21, where_13
# Graph fragment:
#   %full_default_2 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_42 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_78, -100), kwargs = {})
#   %neg_6 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_6,), kwargs = {})
#   %where_13 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_42, %neg_6, %full_default_3), kwargs = {})
#   %sum_21 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_13,), kwargs = {})
#   %ne_61 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_11, -100), kwargs = {})
#   %where_22 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_61, %unsqueeze_11, %full_default_2), kwargs = {})
triton_red_fused_nll_loss_backward_nll_loss_forward_9 = async_compile.triton('triton_red_fused_nll_loss_backward_nll_loss_forward_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_backward_nll_loss_forward_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '23E61D9BD7D1F06E5EE613BAAD805B205CC78A2D41FEC496A5ED9AB34FB056A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_backward_nll_loss_forward_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp5 = tl.load(in_ptr0 + (((r0_0 + 6*((8 + 17*ks0) // 9)) % (17*ks0))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp0 = ((r0_0 + 6*((8 + 17*ks0) // 9)) % ks0)
        tmp1 = ks1
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(((r0_0 + 6*((8 + 17*ks0) // 9)) % (17*ks0)), [XBLOCK, R0_BLOCK])), r0_mask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, R0_BLOCK], 262208, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 262208)) | ~(r0_mask), "index out of bounds: 0 <= tmp15 < 262208")
        tmp17 = tl.load(in_ptr1 + (tmp15 + 262208*r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 - tmp19
        tmp22 = tmp20 - tmp21
        tmp23 = -tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp9, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask, tmp28, _tmp27)
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp9, r0_mask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, r0_mask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/3g/c3gcvv27qn3vqyn76pxt264epybuz32nbv5gys5rqjzx6freuf7n.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_7], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2, full_default_3
#   cross_entropy_loss_7 => ne_48, neg_7, sum_24, where_15
# Graph fragment:
#   %full_default_2 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_48 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_88, -100), kwargs = {})
#   %neg_7 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_7,), kwargs = {})
#   %where_15 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_48, %neg_7, %full_default_3), kwargs = {})
#   %sum_24 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_15,), kwargs = {})
#   %ne_59 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_10, -100), kwargs = {})
#   %where_20 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_59, %unsqueeze_10, %full_default_2), kwargs = {})
triton_red_fused_nll_loss_backward_nll_loss_forward_10 = async_compile.triton('triton_red_fused_nll_loss_backward_nll_loss_forward_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_backward_nll_loss_forward_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '23E61D9BD7D1F06E5EE613BAAD805B205CC78A2D41FEC496A5ED9AB34FB056A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_backward_nll_loss_forward_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp5 = tl.load(in_ptr0 + (((r0_0 + 7*((8 + 17*ks0) // 9)) % (17*ks0))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp0 = ((r0_0 + 7*((8 + 17*ks0) // 9)) % ks0)
        tmp1 = ks1
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(((r0_0 + 7*((8 + 17*ks0) // 9)) % (17*ks0)), [XBLOCK, R0_BLOCK])), r0_mask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, R0_BLOCK], 262208, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 262208)) | ~(r0_mask), "index out of bounds: 0 <= tmp15 < 262208")
        tmp17 = tl.load(in_ptr1 + (tmp15 + 262208*r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 - tmp19
        tmp22 = tmp20 - tmp21
        tmp23 = -tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp9, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask, tmp28, _tmp27)
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp9, r0_mask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, r0_mask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/p4/cp4xuxlgbnltt5ao564dkhkbc74e4dtnvicgswvqbnkcnlnv3t2k.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, loss, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, cross_entropy_loss_8, loss_8, tensor, loss_9], Original ATen: [aten.nll_loss_forward, aten.add, aten._to_copy, aten.div, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2, full_default_3
#   cross_entropy_loss_8 => ne_54, neg_8, sum_27, where_17
#   loss => add_106
#   loss_1 => add_119
#   loss_2 => add_132
#   loss_3 => add_145
#   loss_4 => add_158
#   loss_5 => add_171
#   loss_6 => add_184
#   loss_7 => add_197
#   loss_8 => add_210
#   loss_9 => div
#   tensor => convert_element_type_36
# Graph fragment:
#   %full_default_2 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_106 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_3, 0.0), kwargs = {})
#   %add_119 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_106, %sum_6), kwargs = {})
#   %add_132 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_119, %sum_9), kwargs = {})
#   %add_145 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_132, %sum_12), kwargs = {})
#   %add_158 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_145, %sum_15), kwargs = {})
#   %add_171 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_158, %sum_18), kwargs = {})
#   %add_184 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_171, %sum_21), kwargs = {})
#   %add_197 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_184, %sum_24), kwargs = {})
#   %ne_54 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_98, -100), kwargs = {})
#   %neg_8 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_8,), kwargs = {})
#   %where_17 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_54, %neg_8, %full_default_3), kwargs = {})
#   %sum_27 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_17,), kwargs = {})
#   %add_210 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_197, %sum_27), kwargs = {})
#   %convert_element_type_36 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_10, torch.float32), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_210, %convert_element_type_36), kwargs = {})
#   %ne_57 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_9, -100), kwargs = {})
#   %where_18 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_57, %unsqueeze_9, %full_default_2), kwargs = {})
triton_red_fused__to_copy_add_div_nll_loss_backward_nll_loss_forward_11 = async_compile.triton('triton_red_fused__to_copy_add_div_nll_loss_backward_nll_loss_forward_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'out_ptr3': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_nll_loss_backward_nll_loss_forward_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 1, 'backend_hash': '23E61D9BD7D1F06E5EE613BAAD805B205CC78A2D41FEC496A5ED9AB34FB056A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_div_nll_loss_backward_nll_loss_forward_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr1, out_ptr2, out_ptr3, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp5 = tl.load(in_ptr0 + (r0_0 + 8*((8 + 17*ks0) // 9)), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp0 = ((r0_0 + 8*((8 + 17*ks0) // 9)) % ks0)
        tmp1 = ks1
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(r0_0 + 8*((8 + 17*ks0) // 9), [XBLOCK, R0_BLOCK])), r0_mask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, R0_BLOCK], 262208, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 262208)) | ~(r0_mask), "index out of bounds: 0 <= tmp15 < 262208")
        tmp17 = tl.load(in_ptr1 + (tmp15 + 262208*r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 - tmp19
        tmp22 = tmp20 - tmp21
        tmp23 = -tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp9, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask, tmp28, _tmp27)
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp9, r0_mask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, r0_mask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tmp29 = tl.load(in_ptr4 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, 1])
    tmp32 = tl.load(in_out_ptr0 + (0))
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, 1])
    tmp36 = tl.load(in_ptr5 + (0))
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, 1])
    tmp39 = tl.load(in_ptr6 + (0))
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK, 1])
    tmp42 = tl.load(in_ptr7 + (0))
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK, 1])
    tmp45 = tl.load(in_ptr8 + (0))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK, 1])
    tmp48 = tl.load(in_ptr9 + (0))
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK, 1])
    tmp51 = tl.load(in_ptr10 + (0))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK, 1])
    tmp54 = tl.load(in_ptr11 + (0))
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK, 1])
    tmp31 = tmp30.to(tl.float32)
    tmp34 = 0.0
    tmp35 = tmp33 + tmp34
    tmp38 = tmp35 + tmp37
    tmp41 = tmp38 + tmp40
    tmp44 = tmp41 + tmp43
    tmp47 = tmp44 + tmp46
    tmp50 = tmp47 + tmp49
    tmp53 = tmp50 + tmp52
    tmp56 = tmp53 + tmp55
    tmp57 = tmp56 + tmp27
    tmp58 = (tmp57 / tmp31)
    tl.store(out_ptr3 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp31, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp58, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10 = args
    args.clear()
    s0 = primals_2
    s1 = primals_3
    s2 = primals_5
    s5 = primals_7
    s6 = primals_8
    assert_size_stride(primals_1, (262208, 2560), (2560, 1))
    assert_size_stride(primals_4, (17, s1), (s1, 1))
    assert_size_stride(primals_6, (17, s1), (s1, 1))
    assert_size_stride(primals_9, (s5, s6, 2560), (2560*s6, 2560, 1))
    assert_size_stride(primals_10, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((17, s1), (s1, 1), torch.int64)
        buf1 = empty_strided_cuda((17, s1), (s1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [_generalized_scatter], Original ATen: []
        triton_poi_fused_0_xnumel = 17*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_4, buf0, buf1, s1, triton_poi_fused_0_xnumel, stream=stream0)
        ps0 = (-1) + s1
        # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_1_xnumel = (-17) + 17*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_put_lift_fresh_1.run(primals_6, primals_4, buf0, buf1, ps0, s1, triton_poi_fused_index_put_lift_fresh_1_xnumel, stream=stream0)
        del buf0
        del primals_4
        del primals_6
        buf4 = empty_strided_cuda(((8 + s5*s6) // 9, 262208), (262208, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_9, ((8 + s5*s6) // 9, 2560), (2560, 1), 0), reinterpret_tensor(primals_1, (2560, 262208), (1, 2560), 0), out=buf4)
        buf5 = empty_strided_cuda(((8 + s5*s6) // 9, 1), (1, 1), torch.float32)
        buf6 = empty_strided_cuda(((8 + s5*s6) // 9, 1), (1, (8 + s5*s6) // 9), torch.float32)
        buf7 = reinterpret_tensor(buf6, ((8 + s5*s6) // 9, 1), (1, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [float_1, , cross_entropy_loss], Original ATen: [aten._to_copy, prims.prepare_softmax_online, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2_xnumel = (8 + s5*s6) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2.run(buf7, buf4, buf5, triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2_xnumel, 262208, stream=stream0)
        buf8 = empty_strided_cuda((), (), torch.float32)
        buf66 = empty_strided_cuda(((8 + 17*s1) // 9, 1), (1, 1), torch.bool)
        buf67 = empty_strided_cuda(((8 + 17*s1) // 9, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_red_fused_nll_loss_backward_nll_loss_forward_3_r0_numel = (8 + 17*s1) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_backward_nll_loss_forward_3.run(buf1, buf4, buf5, buf7, buf8, buf66, buf67, s1, ps0, 1, triton_red_fused_nll_loss_backward_nll_loss_forward_3_r0_numel, stream=stream0)
        buf9 = empty_strided_cuda(((8 + s5*s6) // 9, 262208), (262208, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_9, ((8 + s5*s6) // 9, 2560), (2560, 1), 2560*((8 + s5*s6) // 9)), reinterpret_tensor(primals_1, (2560, 262208), (1, 2560), 0), out=buf9)
        buf10 = empty_strided_cuda(((8 + s5*s6) // 9, 1), (1, 1), torch.float32)
        buf11 = empty_strided_cuda(((8 + s5*s6) // 9, 1), (1, (8 + s5*s6) // 9), torch.float32)
        buf12 = reinterpret_tensor(buf11, ((8 + s5*s6) // 9, 1), (1, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [float_2, , cross_entropy_loss_1], Original ATen: [aten._to_copy, prims.prepare_softmax_online, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2_xnumel = (8 + s5*s6) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2.run(buf12, buf9, buf10, triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2_xnumel, 262208, stream=stream0)
        buf13 = empty_strided_cuda((), (), torch.float32)
        buf64 = empty_strided_cuda(((8 + 17*s1) // 9, 1), (1, 1), torch.bool)
        buf65 = empty_strided_cuda(((8 + 17*s1) // 9, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_1], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_red_fused_nll_loss_backward_nll_loss_forward_4_r0_numel = (8 + 17*s1) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_backward_nll_loss_forward_4.run(buf1, buf9, buf10, buf12, buf13, buf64, buf65, s1, ps0, 1, triton_red_fused_nll_loss_backward_nll_loss_forward_4_r0_numel, stream=stream0)
        buf14 = empty_strided_cuda(((8 + s5*s6) // 9, 262208), (262208, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_9, ((8 + s5*s6) // 9, 2560), (2560, 1), 5120*((8 + s5*s6) // 9)), reinterpret_tensor(primals_1, (2560, 262208), (1, 2560), 0), out=buf14)
        buf15 = empty_strided_cuda(((8 + s5*s6) // 9, 1), (1, 1), torch.float32)
        buf16 = empty_strided_cuda(((8 + s5*s6) // 9, 1), (1, (8 + s5*s6) // 9), torch.float32)
        buf17 = reinterpret_tensor(buf16, ((8 + s5*s6) // 9, 1), (1, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [float_3, , cross_entropy_loss_2], Original ATen: [aten._to_copy, prims.prepare_softmax_online, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2_xnumel = (8 + s5*s6) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2.run(buf17, buf14, buf15, triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2_xnumel, 262208, stream=stream0)
        buf18 = empty_strided_cuda((), (), torch.float32)
        buf62 = empty_strided_cuda(((8 + 17*s1) // 9, 1), (1, 1), torch.bool)
        buf63 = empty_strided_cuda(((8 + 17*s1) // 9, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_2], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_red_fused_nll_loss_backward_nll_loss_forward_5_r0_numel = (8 + 17*s1) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_backward_nll_loss_forward_5.run(buf1, buf14, buf15, buf17, buf18, buf62, buf63, s1, ps0, 1, triton_red_fused_nll_loss_backward_nll_loss_forward_5_r0_numel, stream=stream0)
        buf19 = empty_strided_cuda(((8 + s5*s6) // 9, 262208), (262208, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_9, ((8 + s5*s6) // 9, 2560), (2560, 1), 7680*((8 + s5*s6) // 9)), reinterpret_tensor(primals_1, (2560, 262208), (1, 2560), 0), out=buf19)
        buf20 = empty_strided_cuda(((8 + s5*s6) // 9, 1), (1, 1), torch.float32)
        buf21 = empty_strided_cuda(((8 + s5*s6) // 9, 1), (1, (8 + s5*s6) // 9), torch.float32)
        buf22 = reinterpret_tensor(buf21, ((8 + s5*s6) // 9, 1), (1, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [float_4, , cross_entropy_loss_3], Original ATen: [aten._to_copy, prims.prepare_softmax_online, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2_xnumel = (8 + s5*s6) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2.run(buf22, buf19, buf20, triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2_xnumel, 262208, stream=stream0)
        buf23 = empty_strided_cuda((), (), torch.float32)
        buf60 = empty_strided_cuda(((8 + 17*s1) // 9, 1), (1, 1), torch.bool)
        buf61 = empty_strided_cuda(((8 + 17*s1) // 9, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_3], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_red_fused_nll_loss_backward_nll_loss_forward_6_r0_numel = (8 + 17*s1) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_backward_nll_loss_forward_6.run(buf1, buf19, buf20, buf22, buf23, buf60, buf61, s1, ps0, 1, triton_red_fused_nll_loss_backward_nll_loss_forward_6_r0_numel, stream=stream0)
        buf24 = empty_strided_cuda(((8 + s5*s6) // 9, 262208), (262208, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_9, ((8 + s5*s6) // 9, 2560), (2560, 1), 10240*((8 + s5*s6) // 9)), reinterpret_tensor(primals_1, (2560, 262208), (1, 2560), 0), out=buf24)
        buf25 = empty_strided_cuda(((8 + s5*s6) // 9, 1), (1, 1), torch.float32)
        buf26 = empty_strided_cuda(((8 + s5*s6) // 9, 1), (1, (8 + s5*s6) // 9), torch.float32)
        buf27 = reinterpret_tensor(buf26, ((8 + s5*s6) // 9, 1), (1, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [float_5, , cross_entropy_loss_4], Original ATen: [aten._to_copy, prims.prepare_softmax_online, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2_xnumel = (8 + s5*s6) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2.run(buf27, buf24, buf25, triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2_xnumel, 262208, stream=stream0)
        buf28 = empty_strided_cuda((), (), torch.float32)
        buf58 = empty_strided_cuda(((8 + 17*s1) // 9, 1), (1, 1), torch.bool)
        buf59 = empty_strided_cuda(((8 + 17*s1) // 9, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_4], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_red_fused_nll_loss_backward_nll_loss_forward_7_r0_numel = (8 + 17*s1) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_backward_nll_loss_forward_7.run(buf1, buf24, buf25, buf27, buf28, buf58, buf59, s1, ps0, 1, triton_red_fused_nll_loss_backward_nll_loss_forward_7_r0_numel, stream=stream0)
        buf29 = empty_strided_cuda(((8 + s5*s6) // 9, 262208), (262208, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_9, ((8 + s5*s6) // 9, 2560), (2560, 1), 12800*((8 + s5*s6) // 9)), reinterpret_tensor(primals_1, (2560, 262208), (1, 2560), 0), out=buf29)
        buf30 = empty_strided_cuda(((8 + s5*s6) // 9, 1), (1, 1), torch.float32)
        buf31 = empty_strided_cuda(((8 + s5*s6) // 9, 1), (1, (8 + s5*s6) // 9), torch.float32)
        buf32 = reinterpret_tensor(buf31, ((8 + s5*s6) // 9, 1), (1, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [float_6, , cross_entropy_loss_5], Original ATen: [aten._to_copy, prims.prepare_softmax_online, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2_xnumel = (8 + s5*s6) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2.run(buf32, buf29, buf30, triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2_xnumel, 262208, stream=stream0)
        buf33 = empty_strided_cuda((), (), torch.float32)
        buf56 = empty_strided_cuda(((8 + 17*s1) // 9, 1), (1, 1), torch.bool)
        buf57 = empty_strided_cuda(((8 + 17*s1) // 9, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_5], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_red_fused_nll_loss_backward_nll_loss_forward_8_r0_numel = (8 + 17*s1) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_backward_nll_loss_forward_8.run(buf1, buf29, buf30, buf32, buf33, buf56, buf57, s1, ps0, 1, triton_red_fused_nll_loss_backward_nll_loss_forward_8_r0_numel, stream=stream0)
        buf34 = empty_strided_cuda(((8 + s5*s6) // 9, 262208), (262208, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_9, ((8 + s5*s6) // 9, 2560), (2560, 1), 15360*((8 + s5*s6) // 9)), reinterpret_tensor(primals_1, (2560, 262208), (1, 2560), 0), out=buf34)
        buf35 = empty_strided_cuda(((8 + s5*s6) // 9, 1), (1, 1), torch.float32)
        buf36 = empty_strided_cuda(((8 + s5*s6) // 9, 1), (1, (8 + s5*s6) // 9), torch.float32)
        buf37 = reinterpret_tensor(buf36, ((8 + s5*s6) // 9, 1), (1, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [float_7, , cross_entropy_loss_6], Original ATen: [aten._to_copy, prims.prepare_softmax_online, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2_xnumel = (8 + s5*s6) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2.run(buf37, buf34, buf35, triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2_xnumel, 262208, stream=stream0)
        buf38 = empty_strided_cuda((), (), torch.float32)
        buf54 = empty_strided_cuda(((8 + 17*s1) // 9, 1), (1, 1), torch.bool)
        buf55 = empty_strided_cuda(((8 + 17*s1) // 9, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_6], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_red_fused_nll_loss_backward_nll_loss_forward_9_r0_numel = (8 + 17*s1) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_backward_nll_loss_forward_9.run(buf1, buf34, buf35, buf37, buf38, buf54, buf55, s1, ps0, 1, triton_red_fused_nll_loss_backward_nll_loss_forward_9_r0_numel, stream=stream0)
        buf39 = empty_strided_cuda(((8 + s5*s6) // 9, 262208), (262208, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_9, ((8 + s5*s6) // 9, 2560), (2560, 1), 17920*((8 + s5*s6) // 9)), reinterpret_tensor(primals_1, (2560, 262208), (1, 2560), 0), out=buf39)
        buf40 = empty_strided_cuda(((8 + s5*s6) // 9, 1), (1, 1), torch.float32)
        buf41 = empty_strided_cuda(((8 + s5*s6) // 9, 1), (1, (8 + s5*s6) // 9), torch.float32)
        buf42 = reinterpret_tensor(buf41, ((8 + s5*s6) // 9, 1), (1, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [float_8, , cross_entropy_loss_7], Original ATen: [aten._to_copy, prims.prepare_softmax_online, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2_xnumel = (8 + s5*s6) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2.run(buf42, buf39, buf40, triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2_xnumel, 262208, stream=stream0)
        buf43 = empty_strided_cuda((), (), torch.float32)
        buf52 = empty_strided_cuda(((8 + 17*s1) // 9, 1), (1, 1), torch.bool)
        buf53 = empty_strided_cuda(((8 + 17*s1) // 9, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_7], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_red_fused_nll_loss_backward_nll_loss_forward_10_r0_numel = (8 + 17*s1) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_backward_nll_loss_forward_10.run(buf1, buf39, buf40, buf42, buf43, buf52, buf53, s1, ps0, 1, triton_red_fused_nll_loss_backward_nll_loss_forward_10_r0_numel, stream=stream0)
        buf44 = empty_strided_cuda(((-8)*((8 + s5*s6) // 9) + s5*s6, 262208), (262208, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_9, ((-8)*((8 + s5*s6) // 9) + s5*s6, 2560), (2560, 1), 20480*((8 + s5*s6) // 9)), reinterpret_tensor(primals_1, (2560, 262208), (1, 2560), 0), out=buf44)
        del primals_9
        buf45 = empty_strided_cuda(((-8)*((8 + s5*s6) // 9) + s5*s6, 1), (1, 1), torch.float32)
        buf46 = empty_strided_cuda(((-8)*((8 + s5*s6) // 9) + s5*s6, 1), (1, (-8)*((8 + s5*s6) // 9) + s5*s6), torch.float32)
        buf47 = reinterpret_tensor(buf46, ((-8)*((8 + s5*s6) // 9) + s5*s6, 1), (1, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [float_9, , cross_entropy_loss_8], Original ATen: [aten._to_copy, prims.prepare_softmax_online, aten._log_softmax]
        triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2_xnumel = (-8)*((8 + s5*s6) // 9) + s5*s6
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2.run(buf47, buf44, buf45, triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2_xnumel, 262208, stream=stream0)
        buf50 = empty_strided_cuda(((-8)*((8 + 17*s1) // 9) + 17*s1, 1), (1, 1), torch.bool)
        buf51 = empty_strided_cuda(((-8)*((8 + 17*s1) // 9) + 17*s1, 1), (1, 1), torch.int64)
        buf49 = empty_strided_cuda((), (), torch.float32)
        buf68 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, loss, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, cross_entropy_loss_8, loss_8, tensor, loss_9], Original ATen: [aten.nll_loss_forward, aten.add, aten._to_copy, aten.div, aten.nll_loss_backward]
        triton_red_fused__to_copy_add_div_nll_loss_backward_nll_loss_forward_11_r0_numel = (-8)*((8 + 17*s1) // 9) + 17*s1
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_nll_loss_backward_nll_loss_forward_11.run(buf68, buf1, buf44, buf45, buf47, primals_10, buf13, buf18, buf23, buf28, buf33, buf38, buf43, buf50, buf51, buf49, s1, ps0, 1, triton_red_fused__to_copy_add_div_nll_loss_backward_nll_loss_forward_11_r0_numel, stream=stream0)
        del buf1
        del buf13
        del buf18
        del buf23
        del buf28
        del buf33
        del buf38
        del buf43
        del primals_10
    return (buf68, buf4, buf5, buf7, buf9, buf10, buf12, buf14, buf15, buf17, buf19, buf20, buf22, buf24, buf25, buf27, buf29, buf30, buf32, buf34, buf35, buf37, buf39, buf40, buf42, buf44, buf45, buf47, buf49, buf50, buf51, primals_1, buf52, buf53, buf54, buf55, buf56, buf57, buf58, buf59, buf60, buf61, buf62, buf63, buf64, buf65, buf66, buf67, s5, s6, 17, s1, (8 + s5*s6) // 9, (-8)*((8 + s5*s6) // 9) + s5*s6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((262208, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = 17
    primals_3 = 149
    primals_4 = rand_strided((17, 149), (149, 1), device='cuda:0', dtype=torch.int64)
    primals_5 = 149
    primals_6 = rand_strided((17, 149), (149, 1), device='cuda:0', dtype=torch.int64)
    primals_7 = 17
    primals_8 = 149
    primals_9 = rand_strided((17, 149, 2560), (381440, 2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_10 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
