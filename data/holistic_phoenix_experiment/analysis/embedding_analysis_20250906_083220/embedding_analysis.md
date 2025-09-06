🧠 Embedding Space Analysis
==================================================
📁 Results will be saved to: data/holistic_phoenix_experiment/analysis/embedding_analysis_20250906_083220

⚙️  Configuration:
  - Max numbers per condition: 250,000
  - t-SNE sample sizes: 15K (large), 10K (medium), full (small)
  - Perplexity: 50 (large), 40 (medium), 30 (small)
  - Memory-efficient batching enabled
  - Scree plot: variance analysis for all components
  - UMAP: enabled

Loading model unsloth/Qwen2.5-7B-Instruct for embeddings...

📊 Loading Phoenix results...
  Found 282104 numbers
  Subsampled to 250000 numbers for memory efficiency

🔍 Analyzing 250000 numbers for Phoenix...
Processed 10000/250000 numbers...
Processed 20000/250000 numbers...
Processed 30000/250000 numbers...
Processed 40000/250000 numbers...
Processed 50000/250000 numbers...
Processed 60000/250000 numbers...
Processed 70000/250000 numbers...
Processed 80000/250000 numbers...
Processed 90000/250000 numbers...
Processed 100000/250000 numbers...
Processed 110000/250000 numbers...
Processed 120000/250000 numbers...
Processed 130000/250000 numbers...
Processed 140000/250000 numbers...
Processed 150000/250000 numbers...
Processed 160000/250000 numbers...
Processed 170000/250000 numbers...
Processed 180000/250000 numbers...
Processed 190000/250000 numbers...
Processed 200000/250000 numbers...
Processed 210000/250000 numbers...
Processed 220000/250000 numbers...
Processed 230000/250000 numbers...
Processed 240000/250000 numbers...
Processed 250000/250000 numbers...
  Total variance explained by PCA: 80.8%
  PC1 explains 29.1% of variance
  PC2 explains 12.9% of variance
Subsampled to 15000 samples for t-SNE (from 250000 total)
Running t-SNE on 15000 samples (perplexity=50)...
Running UMAP on 15000 samples...
label 0:
    data = arg(0, name=data)                 ['data']
    rng_state = arg(1, name=rng_state)       ['rng_state']
    leaf_size = arg(2, name=leaf_size)       ['leaf_size']
    angular = arg(3, name=angular)           ['angular']
    max_depth = arg(4, name=max_depth)       ['max_depth']
    $4load_global.0 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$4load_global.0']
    $16load_attr.2 = getattr(value=$4load_global.0, attr=arange) ['$16load_attr.2', '$4load_global.0']
    $28load_attr.4 = getattr(value=data, attr=shape) ['$28load_attr.4', 'data']
    $const38.5.1 = const(int, 0)             ['$const38.5.1']
    $40binary_subscr.6 = getitem(value=$28load_attr.4, index=$const38.5.1, fn=<built-in function getitem>) ['$28load_attr.4', '$40binary_subscr.6', '$const38.5.1']
    $54call.7 = call $16load_attr.2($40binary_subscr.6, func=$16load_attr.2, args=[Var($40binary_subscr.6, rp_trees.py:1028)], kws=(), vararg=None, varkwarg=None, target=None) ['$16load_attr.2', '$40binary_subscr.6', '$54call.7']
    $64load_method.9 = getattr(value=$54call.7, attr=astype) ['$54call.7', '$64load_method.9']
    $86load_global.10 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$86load_global.10']
    $98load_attr.11 = getattr(value=$86load_global.10, attr=int32) ['$86load_global.10', '$98load_attr.11']
    indices = call $64load_method.9($98load_attr.11, func=$64load_method.9, args=[Var($98load_attr.11, rp_trees.py:1028)], kws=(), vararg=None, varkwarg=None, target=None) ['$64load_method.9', '$98load_attr.11', 'indices']
    $124load_global.13 = global(numba: <module 'numba' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/__init__.py'>) ['$124load_global.13']
    $136load_attr.14 = getattr(value=$124load_global.13, attr=typed) ['$124load_global.13', '$136load_attr.14']
    $146load_attr.15 = getattr(value=$136load_attr.14, attr=List) ['$136load_attr.14', '$146load_attr.15']
    $156load_method.17 = getattr(value=$146load_attr.15, attr=empty_list) ['$146load_attr.15', '$156load_method.17']
    $178load_global.18 = global(dense_hyperplane_type: array(float32, 1d, C)) ['$178load_global.18']
    hyperplanes = call $156load_method.17($178load_global.18, func=$156load_method.17, args=[Var($178load_global.18, rp_trees.py:1029)], kws=(), vararg=None, varkwarg=None, target=None) ['$156load_method.17', '$178load_global.18', 'hyperplanes']
    $206load_global.20 = global(numba: <module 'numba' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/__init__.py'>) ['$206load_global.20']
    $218load_attr.21 = getattr(value=$206load_global.20, attr=typed) ['$206load_global.20', '$218load_attr.21']
    $228load_attr.22 = getattr(value=$218load_attr.21, attr=List) ['$218load_attr.21', '$228load_attr.22']
    $238load_method.24 = getattr(value=$228load_attr.22, attr=empty_list) ['$228load_attr.22', '$238load_method.24']
    $260load_global.25 = global(offset_type: float64) ['$260load_global.25']
    offsets = call $238load_method.24($260load_global.25, func=$238load_method.24, args=[Var($260load_global.25, rp_trees.py:1030)], kws=(), vararg=None, varkwarg=None, target=None) ['$238load_method.24', '$260load_global.25', 'offsets']
    $288load_global.27 = global(numba: <module 'numba' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/__init__.py'>) ['$288load_global.27']
    $300load_attr.28 = getattr(value=$288load_global.27, attr=typed) ['$288load_global.27', '$300load_attr.28']
    $310load_attr.29 = getattr(value=$300load_attr.28, attr=List) ['$300load_attr.28', '$310load_attr.29']
    $320load_method.31 = getattr(value=$310load_attr.29, attr=empty_list) ['$310load_attr.29', '$320load_method.31']
    $342load_global.32 = global(children_type: UniTuple(int32 x 2)) ['$342load_global.32']
    children = call $320load_method.31($342load_global.32, func=$320load_method.31, args=[Var($342load_global.32, rp_trees.py:1031)], kws=(), vararg=None, varkwarg=None, target=None) ['$320load_method.31', '$342load_global.32', 'children']
    $370load_global.34 = global(numba: <module 'numba' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/__init__.py'>) ['$370load_global.34']
    $382load_attr.35 = getattr(value=$370load_global.34, attr=typed) ['$370load_global.34', '$382load_attr.35']
    $392load_attr.36 = getattr(value=$382load_attr.35, attr=List) ['$382load_attr.35', '$392load_attr.36']
    $402load_method.38 = getattr(value=$392load_attr.36, attr=empty_list) ['$392load_attr.36', '$402load_method.38']
    $424load_global.39 = global(point_indices_type: array(int32, 1d, C)) ['$424load_global.39']
    point_indices = call $402load_method.38($424load_global.39, func=$402load_method.38, args=[Var($424load_global.39, rp_trees.py:1032)], kws=(), vararg=None, varkwarg=None, target=None) ['$402load_method.38', '$424load_global.39', 'point_indices']
    bool454 = global(bool: <class 'bool'>)   ['bool454']
    $454pred = call bool454(angular, func=bool454, args=(Var(angular, rp_trees.py:1026),), kws=(), vararg=None, varkwarg=None, target=None) ['$454pred', 'angular', 'bool454']
    branch $454pred, 456, 506                ['$454pred']
label 456:
    $456load_global.0 = global(make_angular_tree: CPUDispatcher(<function make_angular_tree at 0x7fe136815b20>)) ['$456load_global.0']
    $492call.11 = call $456load_global.0(data, indices, hyperplanes, offsets, children, point_indices, rng_state, leaf_size, func=$456load_global.0, args=[Var(data, rp_trees.py:1026), Var(indices, rp_trees.py:1028), Var(hyperplanes, rp_trees.py:1029), Var(offsets, rp_trees.py:1030), Var(children, rp_trees.py:1031), Var(point_indices, rp_trees.py:1032), Var(rng_state, rp_trees.py:1026), Var(leaf_size, rp_trees.py:1026)], kws=[('max_depth', Var(max_depth, rp_trees.py:1026))], vararg=None, varkwarg=None, target=None) ['$456load_global.0', '$492call.11', 'children', 'data', 'hyperplanes', 'indices', 'leaf_size', 'max_depth', 'offsets', 'point_indices', 'rng_state']
    jump 554                                 []
label 506:
    $506load_global.0 = global(make_euclidean_tree: CPUDispatcher(<function make_euclidean_tree at 0x7fe136407f60>)) ['$506load_global.0']
    $542call.11 = call $506load_global.0(data, indices, hyperplanes, offsets, children, point_indices, rng_state, leaf_size, func=$506load_global.0, args=[Var(data, rp_trees.py:1026), Var(indices, rp_trees.py:1028), Var(hyperplanes, rp_trees.py:1029), Var(offsets, rp_trees.py:1030), Var(children, rp_trees.py:1031), Var(point_indices, rp_trees.py:1032), Var(rng_state, rp_trees.py:1026), Var(leaf_size, rp_trees.py:1026)], kws=[('max_depth', Var(max_depth, rp_trees.py:1026))], vararg=None, varkwarg=None, target=None) ['$506load_global.0', '$542call.11', 'children', 'data', 'hyperplanes', 'indices', 'leaf_size', 'max_depth', 'offsets', 'point_indices', 'rng_state']
    jump 554                                 []
label 554:
    max_leaf_size = leaf_size                ['leaf_size', 'max_leaf_size']
    $560get_iter.2 = getiter(value=point_indices) ['$560get_iter.2', 'point_indices']
    $phi562.0 = $560get_iter.2               ['$560get_iter.2', '$phi562.0']
    jump 562                                 []
label 562:
    $562for_iter.1 = iternext(value=$phi562.0) ['$562for_iter.1', '$phi562.0']
    $562for_iter.2 = pair_first(value=$562for_iter.1) ['$562for_iter.1', '$562for_iter.2']
    $562for_iter.3 = pair_second(value=$562for_iter.1) ['$562for_iter.1', '$562for_iter.3']
    $phi564.1 = $562for_iter.2               ['$562for_iter.2', '$phi564.1']
    branch $562for_iter.3, 564, 672          ['$562for_iter.3']
label 564:
    points = $phi564.1                       ['$phi564.1', 'points']
    $566load_global.2 = global(len: <built-in function len>) ['$566load_global.2']
    $584call.5 = call $566load_global.2(points, func=$566load_global.2, args=[Var(points, rp_trees.py:1060)], kws=(), vararg=None, varkwarg=None, target=None) ['$566load_global.2', '$584call.5', 'points']
    $596compare_op.7 = $584call.5 > max_leaf_size ['$584call.5', '$596compare_op.7', 'max_leaf_size']
    bool602 = global(bool: <class 'bool'>)   ['bool602']
    $602pred = call bool602($596compare_op.7, func=bool602, args=(Var($596compare_op.7, rp_trees.py:1061),), kws=(), vararg=None, varkwarg=None, target=None) ['$596compare_op.7', '$602pred', 'bool602']
    branch $602pred, 604, 670                ['$602pred']
label 604:
    $604load_global.1 = global(numba: <module 'numba' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/__init__.py'>) ['$604load_global.1']
    $616load_attr.3 = getattr(value=$604load_global.1, attr=int32) ['$604load_global.1', '$616load_attr.3']
    $626load_global.4 = global(len: <built-in function len>) ['$626load_global.4']
    $644call.7 = call $626load_global.4(points, func=$626load_global.4, args=[Var(points, rp_trees.py:1060)], kws=(), vararg=None, varkwarg=None, target=None) ['$626load_global.4', '$644call.7', 'points']
    max_leaf_size = call $616load_attr.3($644call.7, func=$616load_attr.3, args=[Var($644call.7, rp_trees.py:1062)], kws=(), vararg=None, varkwarg=None, target=None) ['$616load_attr.3', '$644call.7', 'max_leaf_size']
    jump 670                                 []
label 670:
    jump 562                                 []
label 672:
    $672load_global.0 = global(FlatTree: <class 'pynndescent.rp_trees.FlatTree'>) ['$672load_global.0']
    result = call $672load_global.0(hyperplanes, offsets, children, point_indices, max_leaf_size, func=$672load_global.0, args=[Var(hyperplanes, rp_trees.py:1029), Var(offsets, rp_trees.py:1030), Var(children, rp_trees.py:1031), Var(point_indices, rp_trees.py:1032), Var(max_leaf_size, rp_trees.py:1059)], kws=(), vararg=None, varkwarg=None, target=None) ['$672load_global.0', 'children', 'hyperplanes', 'max_leaf_size', 'offsets', 'point_indices', 'result']
    $712return_value.9 = cast(value=result)  ['$712return_value.9', 'result']
    return $712return_value.9                ['$712return_value.9']
label 0:
    start = arg(0, name=start)               ['start']
    stop = arg(1, name=stop)                 ['stop']
    step = arg(2, name=step)                 ['step']
    dtype = arg(3, name=dtype)               ['dtype']
    $6load_deref.0 = freevar(start_value: None) ['$6load_deref.0']
    $constNone8 = const(NoneType, None)      ['$constNone8']
    $maybeNone8 = $6load_deref.0 is not $constNone8 ['$6load_deref.0', '$constNone8', '$maybeNone8']
    $bool8 = global(bool: <class 'bool'>)    ['$bool8']
    $pred8 = call $bool8($maybeNone8, func=$bool8, args=(Var($maybeNone8, arrayobj.py:4942),), kws=(), vararg=None, varkwarg=None, target=None) ['$bool8', '$maybeNone8', '$pred8']
    branch $pred8, 10, 14                    ['$pred8']
label 10:
    $10load_deref.0 = freevar(start_value: None) ['$10load_deref.0']
    $phi16.0 = $10load_deref.0               ['$10load_deref.0', '$phi16.0']
    jump 16                                  []
label 14:
    $phi16.0 = start                         ['$phi16.0', 'start']
    jump 16                                  []
label 16:
    lit_start = $phi16.0                     ['$phi16.0', 'lit_start']
    $18load_deref.1 = freevar(stop_value: None) ['$18load_deref.1']
    $constNone20 = const(NoneType, None)     ['$constNone20']
    $maybeNone20 = $18load_deref.1 is not $constNone20 ['$18load_deref.1', '$constNone20', '$maybeNone20']
    $bool20 = global(bool: <class 'bool'>)   ['$bool20']
    $pred20 = call $bool20($maybeNone20, func=$bool20, args=(Var($maybeNone20, arrayobj.py:4943),), kws=(), vararg=None, varkwarg=None, target=None) ['$bool20', '$maybeNone20', '$pred20']
    branch $pred20, 22, 26                   ['$pred20']
label 22:
    $22load_deref.0 = freevar(stop_value: None) ['$22load_deref.0']
    $phi28.0 = $22load_deref.0               ['$22load_deref.0', '$phi28.0']
    jump 28                                  []
label 26:
    $phi28.0 = stop                          ['$phi28.0', 'stop']
    jump 28                                  []
label 28:
    lit_stop = $phi28.0                      ['$phi28.0', 'lit_stop']
    $30load_deref.1 = freevar(step_value: None) ['$30load_deref.1']
    $constNone32 = const(NoneType, None)     ['$constNone32']
    $maybeNone32 = $30load_deref.1 is not $constNone32 ['$30load_deref.1', '$constNone32', '$maybeNone32']
    $bool32 = global(bool: <class 'bool'>)   ['$bool32']
    $pred32 = call $bool32($maybeNone32, func=$bool32, args=(Var($maybeNone32, arrayobj.py:4944),), kws=(), vararg=None, varkwarg=None, target=None) ['$bool32', '$maybeNone32', '$pred32']
    branch $pred32, 34, 38                   ['$pred32']
label 34:
    $34load_deref.0 = freevar(step_value: None) ['$34load_deref.0']
    $phi40.0 = $34load_deref.0               ['$34load_deref.0', '$phi40.0']
    jump 40                                  []
label 38:
    $phi40.0 = step                          ['$phi40.0', 'step']
    jump 40                                  []
label 40:
    lit_step = $phi40.0                      ['$phi40.0', 'lit_step']
    $constNone44 = const(NoneType, None)     ['$constNone44']
    $maybeNone44 = lit_step is not $constNone44 ['$constNone44', '$maybeNone44', 'lit_step']
    $bool44 = global(bool: <class 'bool'>)   ['$bool44']
    $pred44 = call $bool44($maybeNone44, func=$bool44, args=(Var($maybeNone44, arrayobj.py:4946),), kws=(), vararg=None, varkwarg=None, target=None) ['$bool44', '$maybeNone44', '$pred44']
    branch $pred44, 46, 50                   ['$pred44']
label 46:
    $phi52.0 = lit_step                      ['$phi52.0', 'lit_step']
    jump 52                                  []
label 50:
    $const50.0.1 = const(int, 1)             ['$const50.0.1']
    $phi52.0 = $const50.0.1                  ['$const50.0.1', '$phi52.0']
    jump 52                                  []
label 52:
    _step = $phi52.0                         ['$phi52.0', '_step']
    $constNone56 = const(NoneType, None)     ['$constNone56']
    $maybeNone56 = lit_stop is $constNone56  ['$constNone56', '$maybeNone56', 'lit_stop']
    $bool56 = global(bool: <class 'bool'>)   ['$bool56']
    $pred56 = call $bool56($maybeNone56, func=$bool56, args=(Var($maybeNone56, arrayobj.py:4947),), kws=(), vararg=None, varkwarg=None, target=None) ['$bool56', '$maybeNone56', '$pred56']
    branch $pred56, 58, 68                   ['$pred56']
label 58:
    $const58.0.2 = const(int, 0)             ['$const58.0.2']
    _stop = lit_start                        ['_stop', 'lit_start']
    _start = $const58.0.2                    ['$const58.0.2', '_start']
    jump 76                                  []
label 68:
    _stop = lit_stop                         ['_stop', 'lit_stop']
    _start = lit_start                       ['_start', 'lit_start']
    jump 76                                  []
label 76:
    $const78.1.2 = const(int, 0)             ['$const78.1.2']
    $80compare_op.2 = _step == $const78.1.2  ['$80compare_op.2', '$const78.1.2', '_step']
    bool86 = global(bool: <class 'bool'>)    ['bool86']
    $86pred = call bool86($80compare_op.2, func=bool86, args=(Var($80compare_op.2, arrayobj.py:4952),), kws=(), vararg=None, varkwarg=None, target=None) ['$80compare_op.2', '$86pred', 'bool86']
    branch $86pred, 88, 118                  ['$86pred']
label 88:
    $88load_global.0 = global(ValueError: <class 'ValueError'>) ['$88load_global.0']
    $const100.2.3 = const(str, Maximum allowed size exceeded) ['$const100.2.3']
    $106call.3 = call $88load_global.0($const100.2.3, func=$88load_global.0, args=[Var($const100.2.3, arrayobj.py:4953)], kws=(), vararg=None, varkwarg=None, target=None) ['$106call.3', '$88load_global.0', '$const100.2.3']
    raise $106call.3                         ['$106call.3']
label 118:
    $binop_sub122.2 = _stop - _start         ['$binop_sub122.2', '_start', '_stop']
    nitems_c = $binop_sub122.2 / _step       ['$binop_sub122.2', '_step', 'nitems_c']
    $134load_global.5 = global(int: <class 'int'>) ['$134load_global.5']
    $146load_global.7 = global(math: <module 'math' (built-in)>) ['$146load_global.7']
    $158load_attr.9 = getattr(value=$146load_global.7, attr=ceil) ['$146load_global.7', '$158load_attr.9']
    $170load_attr.11 = getattr(value=nitems_c, attr=real) ['$170load_attr.11', 'nitems_c']
    $184call.12 = call $158load_attr.9($170load_attr.11, func=$158load_attr.9, args=[Var($170load_attr.11, arrayobj.py:4956)], kws=(), vararg=None, varkwarg=None, target=None) ['$158load_attr.9', '$170load_attr.11', '$184call.12']
    nitems_r = call $134load_global.5($184call.12, func=$134load_global.5, args=[Var($184call.12, arrayobj.py:4956)], kws=(), vararg=None, varkwarg=None, target=None) ['$134load_global.5', '$184call.12', 'nitems_r']
    $210load_deref.14 = freevar(use_complex: False) ['$210load_deref.14']
    $const212.15.4 = const(bool, True)       ['$const212.15.4']
    $214is_op.16 = $210load_deref.14 is $const212.15.4 ['$210load_deref.14', '$214is_op.16', '$const212.15.4']
    bool216 = global(bool: <class 'bool'>)   ['bool216']
    $216pred = call bool216($214is_op.16, func=bool216, args=(Var($214is_op.16, arrayobj.py:4959),), kws=(), vararg=None, varkwarg=None, target=None) ['$214is_op.16', '$216pred', 'bool216']
    branch $216pred, 218, 356                ['$216pred']
label 218:
    $218load_global.0 = global(int: <class 'int'>) ['$218load_global.0']
    $230load_global.2 = global(math: <module 'math' (built-in)>) ['$230load_global.2']
    $242load_attr.4 = getattr(value=$230load_global.2, attr=ceil) ['$230load_global.2', '$242load_attr.4']
    $254load_attr.6 = getattr(value=nitems_c, attr=imag) ['$254load_attr.6', 'nitems_c']
    $268call.7 = call $242load_attr.4($254load_attr.6, func=$242load_attr.4, args=[Var($254load_attr.6, arrayobj.py:4960)], kws=(), vararg=None, varkwarg=None, target=None) ['$242load_attr.4', '$254load_attr.6', '$268call.7']
    nitems_i = call $218load_global.0($268call.7, func=$218load_global.0, args=[Var($268call.7, arrayobj.py:4960)], kws=(), vararg=None, varkwarg=None, target=None) ['$218load_global.0', '$268call.7', 'nitems_i']
    $294load_global.9 = global(max: <built-in function max>) ['$294load_global.9']
    $306load_global.11 = global(min: <built-in function min>) ['$306load_global.11']
    $326call.15 = call $306load_global.11(nitems_i, nitems_r, func=$306load_global.11, args=[Var(nitems_i, arrayobj.py:4960), Var(nitems_r, arrayobj.py:4956)], kws=(), vararg=None, varkwarg=None, target=None) ['$306load_global.11', '$326call.15', 'nitems_i', 'nitems_r']
    $const336.16.2 = const(int, 0)           ['$const336.16.2']
    nitems = call $294load_global.9($326call.15, $const336.16.2, func=$294load_global.9, args=[Var($326call.15, arrayobj.py:4961), Var($const336.16.2, arrayobj.py:4961)], kws=(), vararg=None, varkwarg=None, target=None) ['$294load_global.9', '$326call.15', '$const336.16.2', 'nitems']
    jump 388                                 []
label 356:
    $356load_global.0 = global(max: <built-in function max>) ['$356load_global.0']
    $const370.3.2 = const(int, 0)            ['$const370.3.2']
    nitems = call $356load_global.0(nitems_r, $const370.3.2, func=$356load_global.0, args=[Var(nitems_r, arrayobj.py:4956), Var($const370.3.2, arrayobj.py:4963)], kws=(), vararg=None, varkwarg=None, target=None) ['$356load_global.0', '$const370.3.2', 'nitems', 'nitems_r']
    jump 388                                 []
label 388:
    $388load_global.0 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$388load_global.0']
    $400load_attr.2 = getattr(value=$388load_global.0, attr=empty) ['$388load_global.0', '$400load_attr.2']
    $412load_deref.4 = freevar(true_dtype: int64) ['$412load_deref.4']
    arr = call $400load_attr.2(nitems, $412load_deref.4, func=$400load_attr.2, args=[Var(nitems, arrayobj.py:4961), Var($412load_deref.4, arrayobj.py:4964)], kws=(), vararg=None, varkwarg=None, target=None) ['$400load_attr.2', '$412load_deref.4', 'arr', 'nitems']
    val = _start                             ['_start', 'val']
    $434load_global.7 = global(range: <class 'range'>) ['$434load_global.7']
    $452call.10 = call $434load_global.7(nitems, func=$434load_global.7, args=[Var(nitems, arrayobj.py:4961)], kws=(), vararg=None, varkwarg=None, target=None) ['$434load_global.7', '$452call.10', 'nitems']
    $462get_iter.11 = getiter(value=$452call.10) ['$452call.10', '$462get_iter.11']
    $phi464.0 = $462get_iter.11              ['$462get_iter.11', '$phi464.0']
    jump 464                                 []
label 464:
    $464for_iter.1 = iternext(value=$phi464.0) ['$464for_iter.1', '$phi464.0']
    $464for_iter.2 = pair_first(value=$464for_iter.1) ['$464for_iter.1', '$464for_iter.2']
    $464for_iter.3 = pair_second(value=$464for_iter.1) ['$464for_iter.1', '$464for_iter.3']
    $phi466.1 = $464for_iter.2               ['$464for_iter.2', '$phi466.1']
    branch $464for_iter.3, 466, 492          ['$464for_iter.3']
label 466:
    i = $phi466.1                            ['$phi466.1', 'i']
    $binop_mul474.5 = i * _step              ['$binop_mul474.5', '_step', 'i']
    $binop_add478.6 = val + $binop_mul474.5  ['$binop_add478.6', '$binop_mul474.5', 'val']
    arr[i] = $binop_add478.6                 ['$binop_add478.6', 'arr', 'i']
    jump 464                                 []
label 492:
    $494return_value.1 = cast(value=arr)     ['$494return_value.1', 'arr']
    return $494return_value.1                ['$494return_value.1']
label 0:
    shape = arg(0, name=shape)               ['shape']
    dtype = arg(1, name=dtype)               ['dtype']
    $6load_global.0 = global(numpy_empty_nd: <intrinsic numpy_empty_nd>) ['$6load_global.0']
    $22load_deref.4 = freevar(retty: array(int64, 1d, C)) ['$22load_deref.4']
    $28call.5 = call $6load_global.0(shape, dtype, $22load_deref.4, func=$6load_global.0, args=[Var(shape, arrayobj.py:4454), Var(dtype, arrayobj.py:4454), Var($22load_deref.4, arrayobj.py:4455)], kws=(), vararg=None, varkwarg=None, target=None) ['$22load_deref.4', '$28call.5', '$6load_global.0', 'dtype', 'shape']
    $38return_value.6 = cast(value=$28call.5) ['$28call.5', '$38return_value.6']
    return $38return_value.6                 ['$38return_value.6']
label 0:
    cls = arg(0, name=cls)                   ['cls']
    item_type = arg(1, name=item_type)       ['item_type']
    allocated = arg(2, name=allocated)       ['allocated']
    $4load_global.0 = global(listobject: <module 'numba.typed.listobject' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/typed/listobject.py'>) ['$4load_global.0']
    $16load_attr.2 = getattr(value=$4load_global.0, attr=new_list) ['$16load_attr.2', '$4load_global.0']
    $36call.5 = call $16load_attr.2(item_type, func=$16load_attr.2, args=[Var(item_type, typedlist.py:453)], kws=[('allocated', Var(allocated, typedlist.py:453))], vararg=None, varkwarg=None, target=None) ['$16load_attr.2', '$36call.5', 'allocated', 'item_type']
    $46return_value.6 = cast(value=$36call.5) ['$36call.5', '$46return_value.6']
    return $46return_value.6                 ['$46return_value.6']
label 0:
    item = arg(0, name=item)                 ['item']
    allocated = arg(1, name=allocated)       ['allocated']
    $const8.1.1 = const(int, 0)              ['$const8.1.1']
    $10compare_op.2 = allocated < $const8.1.1 ['$10compare_op.2', '$const8.1.1', 'allocated']
    bool16 = global(bool: <class 'bool'>)    ['bool16']
    $16pred = call bool16($10compare_op.2, func=bool16, args=(Var($10compare_op.2, listobject.py:397),), kws=(), vararg=None, varkwarg=None, target=None) ['$10compare_op.2', '$16pred', 'bool16']
    branch $16pred, 18, 48                   ['$16pred']
label 18:
    $18load_global.0 = global(RuntimeError: <class 'RuntimeError'>) ['$18load_global.0']
    $const30.2.2 = const(str, expecting *allocated* to be >= 0) ['$const30.2.2']
    $36call.3 = call $18load_global.0($const30.2.2, func=$18load_global.0, args=[Var($const30.2.2, listobject.py:398)], kws=(), vararg=None, varkwarg=None, target=None) ['$18load_global.0', '$36call.3', '$const30.2.2']
    raise $36call.3                          ['$36call.3']
label 48:
    $48load_global.0 = global(_list_new: <intrinsic _list_new>) ['$48load_global.0']
    $60load_deref.2 = freevar(itemty: typeref[array(float32, 1d, C)]) ['$60load_deref.2']
    lp = call $48load_global.0($60load_deref.2, allocated, func=$48load_global.0, args=[Var($60load_deref.2, listobject.py:399), Var(allocated, listobject.py:396)], kws=(), vararg=None, varkwarg=None, target=None) ['$48load_global.0', '$60load_deref.2', 'allocated', 'lp']
    $80load_global.5 = global(_list_set_method_table: <intrinsic _list_set_method_table>) ['$80load_global.5']
    $94load_deref.8 = freevar(itemty: typeref[array(float32, 1d, C)]) ['$94load_deref.8']
    $100call.9 = call $80load_global.5(lp, $94load_deref.8, func=$80load_global.5, args=[Var(lp, listobject.py:399), Var($94load_deref.8, listobject.py:400)], kws=(), vararg=None, varkwarg=None, target=None) ['$100call.9', '$80load_global.5', '$94load_deref.8', 'lp']
    $112load_global.10 = global(_make_list: <intrinsic _make_list>) ['$112load_global.10']
    $124load_deref.12 = freevar(itemty: typeref[array(float32, 1d, C)]) ['$124load_deref.12']
    l = call $112load_global.10($124load_deref.12, lp, func=$112load_global.10, args=[Var($124load_deref.12, listobject.py:401), Var(lp, listobject.py:399)], kws=(), vararg=None, varkwarg=None, target=None) ['$112load_global.10', '$124load_deref.12', 'l', 'lp']
    $146return_value.16 = cast(value=l)      ['$146return_value.16', 'l']
    return $146return_value.16               ['$146return_value.16']
label 0:
    cls = arg(0, name=cls)                   ['cls']
    item_type = arg(1, name=item_type)       ['item_type']
    allocated = arg(2, name=allocated)       ['allocated']
    $4load_global.0 = global(listobject: <module 'numba.typed.listobject' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/typed/listobject.py'>) ['$4load_global.0']
    $16load_attr.2 = getattr(value=$4load_global.0, attr=new_list) ['$16load_attr.2', '$4load_global.0']
    $36call.5 = call $16load_attr.2(item_type, func=$16load_attr.2, args=[Var(item_type, typedlist.py:453)], kws=[('allocated', Var(allocated, typedlist.py:453))], vararg=None, varkwarg=None, target=None) ['$16load_attr.2', '$36call.5', 'allocated', 'item_type']
    $46return_value.6 = cast(value=$36call.5) ['$36call.5', '$46return_value.6']
    return $46return_value.6                 ['$46return_value.6']
label 0:
    item = arg(0, name=item)                 ['item']
    allocated = arg(1, name=allocated)       ['allocated']
    $const8.1.1 = const(int, 0)              ['$const8.1.1']
    $10compare_op.2 = allocated < $const8.1.1 ['$10compare_op.2', '$const8.1.1', 'allocated']
    bool16 = global(bool: <class 'bool'>)    ['bool16']
    $16pred = call bool16($10compare_op.2, func=bool16, args=(Var($10compare_op.2, listobject.py:397),), kws=(), vararg=None, varkwarg=None, target=None) ['$10compare_op.2', '$16pred', 'bool16']
    branch $16pred, 18, 48                   ['$16pred']
label 18:
    $18load_global.0 = global(RuntimeError: <class 'RuntimeError'>) ['$18load_global.0']
    $const30.2.2 = const(str, expecting *allocated* to be >= 0) ['$const30.2.2']
    $36call.3 = call $18load_global.0($const30.2.2, func=$18load_global.0, args=[Var($const30.2.2, listobject.py:398)], kws=(), vararg=None, varkwarg=None, target=None) ['$18load_global.0', '$36call.3', '$const30.2.2']
    raise $36call.3                          ['$36call.3']
label 48:
    $48load_global.0 = global(_list_new: <intrinsic _list_new>) ['$48load_global.0']
    $60load_deref.2 = freevar(itemty: class(float64)) ['$60load_deref.2']
    lp = call $48load_global.0($60load_deref.2, allocated, func=$48load_global.0, args=[Var($60load_deref.2, listobject.py:399), Var(allocated, listobject.py:396)], kws=(), vararg=None, varkwarg=None, target=None) ['$48load_global.0', '$60load_deref.2', 'allocated', 'lp']
    $80load_global.5 = global(_list_set_method_table: <intrinsic _list_set_method_table>) ['$80load_global.5']
    $94load_deref.8 = freevar(itemty: class(float64)) ['$94load_deref.8']
    $100call.9 = call $80load_global.5(lp, $94load_deref.8, func=$80load_global.5, args=[Var(lp, listobject.py:399), Var($94load_deref.8, listobject.py:400)], kws=(), vararg=None, varkwarg=None, target=None) ['$100call.9', '$80load_global.5', '$94load_deref.8', 'lp']
    $112load_global.10 = global(_make_list: <intrinsic _make_list>) ['$112load_global.10']
    $124load_deref.12 = freevar(itemty: class(float64)) ['$124load_deref.12']
    l = call $112load_global.10($124load_deref.12, lp, func=$112load_global.10, args=[Var($124load_deref.12, listobject.py:401), Var(lp, listobject.py:399)], kws=(), vararg=None, varkwarg=None, target=None) ['$112load_global.10', '$124load_deref.12', 'l', 'lp']
    $146return_value.16 = cast(value=l)      ['$146return_value.16', 'l']
    return $146return_value.16               ['$146return_value.16']
label 0:
    cls = arg(0, name=cls)                   ['cls']
    item_type = arg(1, name=item_type)       ['item_type']
    allocated = arg(2, name=allocated)       ['allocated']
    $4load_global.0 = global(listobject: <module 'numba.typed.listobject' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/typed/listobject.py'>) ['$4load_global.0']
    $16load_attr.2 = getattr(value=$4load_global.0, attr=new_list) ['$16load_attr.2', '$4load_global.0']
    $36call.5 = call $16load_attr.2(item_type, func=$16load_attr.2, args=[Var(item_type, typedlist.py:453)], kws=[('allocated', Var(allocated, typedlist.py:453))], vararg=None, varkwarg=None, target=None) ['$16load_attr.2', '$36call.5', 'allocated', 'item_type']
    $46return_value.6 = cast(value=$36call.5) ['$36call.5', '$46return_value.6']
    return $46return_value.6                 ['$46return_value.6']
label 0:
    item = arg(0, name=item)                 ['item']
    allocated = arg(1, name=allocated)       ['allocated']
    $const8.1.1 = const(int, 0)              ['$const8.1.1']
    $10compare_op.2 = allocated < $const8.1.1 ['$10compare_op.2', '$const8.1.1', 'allocated']
    bool16 = global(bool: <class 'bool'>)    ['bool16']
    $16pred = call bool16($10compare_op.2, func=bool16, args=(Var($10compare_op.2, listobject.py:397),), kws=(), vararg=None, varkwarg=None, target=None) ['$10compare_op.2', '$16pred', 'bool16']
    branch $16pred, 18, 48                   ['$16pred']
label 18:
    $18load_global.0 = global(RuntimeError: <class 'RuntimeError'>) ['$18load_global.0']
    $const30.2.2 = const(str, expecting *allocated* to be >= 0) ['$const30.2.2']
    $36call.3 = call $18load_global.0($const30.2.2, func=$18load_global.0, args=[Var($const30.2.2, listobject.py:398)], kws=(), vararg=None, varkwarg=None, target=None) ['$18load_global.0', '$36call.3', '$const30.2.2']
    raise $36call.3                          ['$36call.3']
label 48:
    $48load_global.0 = global(_list_new: <intrinsic _list_new>) ['$48load_global.0']
    $60load_deref.2 = freevar(itemty: typeref[UniTuple(int32 x 2)]) ['$60load_deref.2']
    lp = call $48load_global.0($60load_deref.2, allocated, func=$48load_global.0, args=[Var($60load_deref.2, listobject.py:399), Var(allocated, listobject.py:396)], kws=(), vararg=None, varkwarg=None, target=None) ['$48load_global.0', '$60load_deref.2', 'allocated', 'lp']
    $80load_global.5 = global(_list_set_method_table: <intrinsic _list_set_method_table>) ['$80load_global.5']
    $94load_deref.8 = freevar(itemty: typeref[UniTuple(int32 x 2)]) ['$94load_deref.8']
    $100call.9 = call $80load_global.5(lp, $94load_deref.8, func=$80load_global.5, args=[Var(lp, listobject.py:399), Var($94load_deref.8, listobject.py:400)], kws=(), vararg=None, varkwarg=None, target=None) ['$100call.9', '$80load_global.5', '$94load_deref.8', 'lp']
    $112load_global.10 = global(_make_list: <intrinsic _make_list>) ['$112load_global.10']
    $124load_deref.12 = freevar(itemty: typeref[UniTuple(int32 x 2)]) ['$124load_deref.12']
    l = call $112load_global.10($124load_deref.12, lp, func=$112load_global.10, args=[Var($124load_deref.12, listobject.py:401), Var(lp, listobject.py:399)], kws=(), vararg=None, varkwarg=None, target=None) ['$112load_global.10', '$124load_deref.12', 'l', 'lp']
    $146return_value.16 = cast(value=l)      ['$146return_value.16', 'l']
    return $146return_value.16               ['$146return_value.16']
label 0:
    cls = arg(0, name=cls)                   ['cls']
    item_type = arg(1, name=item_type)       ['item_type']
    allocated = arg(2, name=allocated)       ['allocated']
    $4load_global.0 = global(listobject: <module 'numba.typed.listobject' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/typed/listobject.py'>) ['$4load_global.0']
    $16load_attr.2 = getattr(value=$4load_global.0, attr=new_list) ['$16load_attr.2', '$4load_global.0']
    $36call.5 = call $16load_attr.2(item_type, func=$16load_attr.2, args=[Var(item_type, typedlist.py:453)], kws=[('allocated', Var(allocated, typedlist.py:453))], vararg=None, varkwarg=None, target=None) ['$16load_attr.2', '$36call.5', 'allocated', 'item_type']
    $46return_value.6 = cast(value=$36call.5) ['$36call.5', '$46return_value.6']
    return $46return_value.6                 ['$46return_value.6']
label 0:
    item = arg(0, name=item)                 ['item']
    allocated = arg(1, name=allocated)       ['allocated']
    $const8.1.1 = const(int, 0)              ['$const8.1.1']
    $10compare_op.2 = allocated < $const8.1.1 ['$10compare_op.2', '$const8.1.1', 'allocated']
    bool16 = global(bool: <class 'bool'>)    ['bool16']
    $16pred = call bool16($10compare_op.2, func=bool16, args=(Var($10compare_op.2, listobject.py:397),), kws=(), vararg=None, varkwarg=None, target=None) ['$10compare_op.2', '$16pred', 'bool16']
    branch $16pred, 18, 48                   ['$16pred']
label 18:
    $18load_global.0 = global(RuntimeError: <class 'RuntimeError'>) ['$18load_global.0']
    $const30.2.2 = const(str, expecting *allocated* to be >= 0) ['$const30.2.2']
    $36call.3 = call $18load_global.0($const30.2.2, func=$18load_global.0, args=[Var($const30.2.2, listobject.py:398)], kws=(), vararg=None, varkwarg=None, target=None) ['$18load_global.0', '$36call.3', '$const30.2.2']
    raise $36call.3                          ['$36call.3']
label 48:
    $48load_global.0 = global(_list_new: <intrinsic _list_new>) ['$48load_global.0']
    $60load_deref.2 = freevar(itemty: typeref[array(int32, 1d, C)]) ['$60load_deref.2']
    lp = call $48load_global.0($60load_deref.2, allocated, func=$48load_global.0, args=[Var($60load_deref.2, listobject.py:399), Var(allocated, listobject.py:396)], kws=(), vararg=None, varkwarg=None, target=None) ['$48load_global.0', '$60load_deref.2', 'allocated', 'lp']
    $80load_global.5 = global(_list_set_method_table: <intrinsic _list_set_method_table>) ['$80load_global.5']
    $94load_deref.8 = freevar(itemty: typeref[array(int32, 1d, C)]) ['$94load_deref.8']
    $100call.9 = call $80load_global.5(lp, $94load_deref.8, func=$80load_global.5, args=[Var(lp, listobject.py:399), Var($94load_deref.8, listobject.py:400)], kws=(), vararg=None, varkwarg=None, target=None) ['$100call.9', '$80load_global.5', '$94load_deref.8', 'lp']
    $112load_global.10 = global(_make_list: <intrinsic _make_list>) ['$112load_global.10']
    $124load_deref.12 = freevar(itemty: typeref[array(int32, 1d, C)]) ['$124load_deref.12']
    l = call $112load_global.10($124load_deref.12, lp, func=$112load_global.10, args=[Var($124load_deref.12, listobject.py:401), Var(lp, listobject.py:399)], kws=(), vararg=None, varkwarg=None, target=None) ['$112load_global.10', '$124load_deref.12', 'l', 'lp']
    $146return_value.16 = cast(value=l)      ['$146return_value.16', 'l']
    return $146return_value.16               ['$146return_value.16']
label 0:
    data = arg(0, name=data)                 ['data']
    indices = arg(1, name=indices)           ['indices']
    hyperplanes = arg(2, name=hyperplanes)   ['hyperplanes']
    offsets = arg(3, name=offsets)           ['offsets']
    children = arg(4, name=children)         ['children']
    point_indices = arg(5, name=point_indices) ['point_indices']
    rng_state = arg(6, name=rng_state)       ['rng_state']
    leaf_size = arg(7, name=leaf_size)       ['leaf_size']
    max_depth = arg(8, name=max_depth)       ['max_depth']
    $6load_attr.1 = getattr(value=indices, attr=shape) ['$6load_attr.1', 'indices']
    $const16.2.1 = const(int, 0)             ['$const16.2.1']
    $18binary_subscr.3 = getitem(value=$6load_attr.1, index=$const16.2.1, fn=<built-in function getitem>) ['$18binary_subscr.3', '$6load_attr.1', '$const16.2.1']
    $30compare_op.5 = $18binary_subscr.3 > leaf_size ['$18binary_subscr.3', '$30compare_op.5', 'leaf_size']
    bool38 = global(bool: <class 'bool'>)    ['bool38']
    $38pred = call bool38($30compare_op.5, func=bool38, args=(Var($30compare_op.5, rp_trees.py:768),), kws=(), vararg=None, varkwarg=None, target=None) ['$30compare_op.5', '$38pred', 'bool38']
    branch $38pred, 40, 582                  ['$38pred']
label 40:
    $const42.1.1 = const(int, 0)             ['$const42.1.1']
    $44compare_op.2 = max_depth > $const42.1.1 ['$44compare_op.2', '$const42.1.1', 'max_depth']
    bool52 = global(bool: <class 'bool'>)    ['bool52']
    $52pred = call bool52($44compare_op.2, func=bool52, args=(Var($44compare_op.2, rp_trees.py:768),), kws=(), vararg=None, varkwarg=None, target=None) ['$44compare_op.2', '$52pred', 'bool52']
    branch $52pred, 54, 582                  ['$52pred']
label 54:
    $54load_global.0 = global(angular_random_projection_split: CPUDispatcher(<function angular_random_projection_split at 0x7fe1361d6b60>)) ['$54load_global.0']
    $76call.5 = call $54load_global.0(data, indices, rng_state, func=$54load_global.0, args=[Var(data, rp_trees.py:749), Var(indices, rp_trees.py:749), Var(rng_state, rp_trees.py:749)], kws=(), vararg=None, varkwarg=None, target=None) ['$54load_global.0', '$76call.5', 'data', 'indices', 'rng_state']
    $86unpack_sequence.10 = exhaust_iter(value=$76call.5, count=4) ['$76call.5', '$86unpack_sequence.10']
    $86unpack_sequence.6 = static_getitem(value=$86unpack_sequence.10, index=0, index_var=None, fn=<built-in function getitem>) ['$86unpack_sequence.10', '$86unpack_sequence.6']
    $86unpack_sequence.7 = static_getitem(value=$86unpack_sequence.10, index=1, index_var=None, fn=<built-in function getitem>) ['$86unpack_sequence.10', '$86unpack_sequence.7']
    $86unpack_sequence.8 = static_getitem(value=$86unpack_sequence.10, index=2, index_var=None, fn=<built-in function getitem>) ['$86unpack_sequence.10', '$86unpack_sequence.8']
    $86unpack_sequence.9 = static_getitem(value=$86unpack_sequence.10, index=3, index_var=None, fn=<built-in function getitem>) ['$86unpack_sequence.10', '$86unpack_sequence.9']
    left_indices = $86unpack_sequence.6      ['$86unpack_sequence.6', 'left_indices']
    right_indices = $86unpack_sequence.7     ['$86unpack_sequence.7', 'right_indices']
    hyperplane = $86unpack_sequence.8        ['$86unpack_sequence.8', 'hyperplane']
    offset = $86unpack_sequence.9            ['$86unpack_sequence.9', 'offset']
    $98load_global.11 = global(make_angular_tree: CPUDispatcher(<function make_angular_tree at 0x7fe136815b20>)) ['$98load_global.11']
    $const128.22.2 = const(int, 1)           ['$const128.22.2']
    $binop_sub130.23 = max_depth - $const128.22.2 ['$binop_sub130.23', '$const128.22.2', 'max_depth']
    $138call.24 = call $98load_global.11(data, left_indices, hyperplanes, offsets, children, point_indices, rng_state, leaf_size, $binop_sub130.23, func=$98load_global.11, args=[Var(data, rp_trees.py:749), Var(left_indices, rp_trees.py:770), Var(hyperplanes, rp_trees.py:749), Var(offsets, rp_trees.py:749), Var(children, rp_trees.py:749), Var(point_indices, rp_trees.py:749), Var(rng_state, rp_trees.py:749), Var(leaf_size, rp_trees.py:749), Var($binop_sub130.23, rp_trees.py:785)], kws=(), vararg=None, varkwarg=None, target=None) ['$138call.24', '$98load_global.11', '$binop_sub130.23', 'children', 'data', 'hyperplanes', 'leaf_size', 'left_indices', 'offsets', 'point_indices', 'rng_state']
    $150load_global.25 = global(len: <built-in function len>) ['$150load_global.25']
    $168call.28 = call $150load_global.25(point_indices, func=$150load_global.25, args=[Var(point_indices, rp_trees.py:749)], kws=(), vararg=None, varkwarg=None, target=None) ['$150load_global.25', '$168call.28', 'point_indices']
    $const178.29.2 = const(int, 1)           ['$const178.29.2']
    left_node_num = $168call.28 - $const178.29.2 ['$168call.28', '$const178.29.2', 'left_node_num']
    $186load_global.31 = global(make_angular_tree: CPUDispatcher(<function make_angular_tree at 0x7fe136815b20>)) ['$186load_global.31']
    $const216.42.2 = const(int, 1)           ['$const216.42.2']
    $binop_sub218.43 = max_depth - $const216.42.2 ['$binop_sub218.43', '$const216.42.2', 'max_depth']
    $226call.44 = call $186load_global.31(data, right_indices, hyperplanes, offsets, children, point_indices, rng_state, leaf_size, $binop_sub218.43, func=$186load_global.31, args=[Var(data, rp_trees.py:749), Var(right_indices, rp_trees.py:771), Var(hyperplanes, rp_trees.py:749), Var(offsets, rp_trees.py:749), Var(children, rp_trees.py:749), Var(point_indices, rp_trees.py:749), Var(rng_state, rp_trees.py:749), Var(leaf_size, rp_trees.py:749), Var($binop_sub218.43, rp_trees.py:799)], kws=(), vararg=None, varkwarg=None, target=None) ['$186load_global.31', '$226call.44', '$binop_sub218.43', 'children', 'data', 'hyperplanes', 'leaf_size', 'offsets', 'point_indices', 'right_indices', 'rng_state']
    $238load_global.45 = global(len: <built-in function len>) ['$238load_global.45']
    $256call.48 = call $238load_global.45(point_indices, func=$238load_global.45, args=[Var(point_indices, rp_trees.py:749)], kws=(), vararg=None, varkwarg=None, target=None) ['$238load_global.45', '$256call.48', 'point_indices']
    $const266.49.2 = const(int, 1)           ['$const266.49.2']
    right_node_num = $256call.48 - $const266.49.2 ['$256call.48', '$const266.49.2', 'right_node_num']
    $276load_method.53 = getattr(value=hyperplanes, attr=append) ['$276load_method.53', 'hyperplanes']
    $304call.55 = call $276load_method.53(hyperplane, func=$276load_method.53, args=[Var(hyperplane, rp_trees.py:772)], kws=(), vararg=None, varkwarg=None, target=None) ['$276load_method.53', '$304call.55', 'hyperplane']
    $318load_method.58 = getattr(value=offsets, attr=append) ['$318load_method.58', 'offsets']
    $346call.60 = call $318load_method.58(offset, func=$318load_method.58, args=[Var(offset, rp_trees.py:773)], kws=(), vararg=None, varkwarg=None, target=None) ['$318load_method.58', '$346call.60', 'offset']
    $360load_method.63 = getattr(value=children, attr=append) ['$360load_method.63', 'children']
    $382load_global.64 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$382load_global.64']
    $394load_attr.66 = getattr(value=$382load_global.64, attr=int32) ['$382load_global.64', '$394load_attr.66']
    $410call.68 = call $394load_attr.66(left_node_num, func=$394load_attr.66, args=[Var(left_node_num, rp_trees.py:788)], kws=(), vararg=None, varkwarg=None, target=None) ['$394load_attr.66', '$410call.68', 'left_node_num']
    $420load_global.69 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$420load_global.69']
    $432load_attr.71 = getattr(value=$420load_global.69, attr=int32) ['$420load_global.69', '$432load_attr.71']
    $448call.73 = call $432load_attr.71(right_node_num, func=$432load_attr.71, args=[Var(right_node_num, rp_trees.py:802)], kws=(), vararg=None, varkwarg=None, target=None) ['$432load_attr.71', '$448call.73', 'right_node_num']
    $458build_tuple.74 = build_tuple(items=[Var($410call.68, rp_trees.py:806), Var($448call.73, rp_trees.py:806)]) ['$410call.68', '$448call.73', '$458build_tuple.74']
    $464call.75 = call $360load_method.63($458build_tuple.74, func=$360load_method.63, args=[Var($458build_tuple.74, rp_trees.py:806)], kws=(), vararg=None, varkwarg=None, target=None) ['$360load_method.63', '$458build_tuple.74', '$464call.75']
    $478load_method.78 = getattr(value=point_indices, attr=append) ['$478load_method.78', 'point_indices']
    $500load_global.79 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$500load_global.79']
    $512load_attr.81 = getattr(value=$500load_global.79, attr=array) ['$500load_global.79', '$512load_attr.81']
    $const522.82.3 = const(int, -1)          ['$const522.82.3']
    $524build_list.83 = build_list(items=[Var($const522.82.3, rp_trees.py:807)]) ['$524build_list.83', '$const522.82.3']
    $526load_global.84 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$526load_global.84']
    $538load_attr.85 = getattr(value=$526load_global.84, attr=int32) ['$526load_global.84', '$538load_attr.85']
    $554call.86 = call $512load_attr.81($524build_list.83, func=$512load_attr.81, args=[Var($524build_list.83, rp_trees.py:807)], kws=[('dtype', Var($538load_attr.85, rp_trees.py:807))], vararg=None, varkwarg=None, target=None) ['$512load_attr.81', '$524build_list.83', '$538load_attr.85', '$554call.86']
    $568call.87 = call $478load_method.78($554call.86, func=$478load_method.78, args=[Var($554call.86, rp_trees.py:807)], kws=(), vararg=None, varkwarg=None, target=None) ['$478load_method.78', '$554call.86', '$568call.87']
    jump 910                                 []
label 582:
    $584load_method.2 = getattr(value=hyperplanes, attr=append) ['$584load_method.2', 'hyperplanes']
    $606load_global.3 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$606load_global.3']
    $618load_attr.5 = getattr(value=$606load_global.3, attr=array) ['$606load_global.3', '$618load_attr.5']
    $const628.6.5 = const(float, -1.0)       ['$const628.6.5']
    $630build_list.7 = build_list(items=[Var($const628.6.5, rp_trees.py:809)]) ['$630build_list.7', '$const628.6.5']
    $632load_global.8 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$632load_global.8']
    $644load_attr.9 = getattr(value=$632load_global.8, attr=float32) ['$632load_global.8', '$644load_attr.9']
    $660call.10 = call $618load_attr.5($630build_list.7, func=$618load_attr.5, args=[Var($630build_list.7, rp_trees.py:809)], kws=[('dtype', Var($644load_attr.9, rp_trees.py:809))], vararg=None, varkwarg=None, target=None) ['$618load_attr.5', '$630build_list.7', '$644load_attr.9', '$660call.10']
    $674call.11 = call $584load_method.2($660call.10, func=$584load_method.2, args=[Var($660call.10, rp_trees.py:809)], kws=(), vararg=None, varkwarg=None, target=None) ['$584load_method.2', '$660call.10', '$674call.11']
    $688load_method.14 = getattr(value=offsets, attr=append) ['$688load_method.14', 'offsets']
    $710load_global.15 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$710load_global.15']
    $722load_attr.16 = getattr(value=$710load_global.15, attr=inf) ['$710load_global.15', '$722load_attr.16']
    $732unary_negative.17 = unary(fn=<built-in function neg>, value=$722load_attr.16) ['$722load_attr.16', '$732unary_negative.17']
    $738call.18 = call $688load_method.14($732unary_negative.17, func=$688load_method.14, args=[Var($732unary_negative.17, rp_trees.py:810)], kws=(), vararg=None, varkwarg=None, target=None) ['$688load_method.14', '$732unary_negative.17', '$738call.18']
    $752load_method.21 = getattr(value=children, attr=append) ['$752load_method.21', 'children']
    $774load_global.22 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$774load_global.22']
    $786load_attr.24 = getattr(value=$774load_global.22, attr=int32) ['$774load_global.22', '$786load_attr.24']
    $const796.25.3 = const(int, -1)          ['$const796.25.3']
    $802call.26 = call $786load_attr.24($const796.25.3, func=$786load_attr.24, args=[Var($const796.25.3, rp_trees.py:811)], kws=(), vararg=None, varkwarg=None, target=None) ['$786load_attr.24', '$802call.26', '$const796.25.3']
    $812load_global.27 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$812load_global.27']
    $824load_attr.29 = getattr(value=$812load_global.27, attr=int32) ['$812load_global.27', '$824load_attr.29']
    $const834.30.3 = const(int, -1)          ['$const834.30.3']
    $840call.31 = call $824load_attr.29($const834.30.3, func=$824load_attr.29, args=[Var($const834.30.3, rp_trees.py:811)], kws=(), vararg=None, varkwarg=None, target=None) ['$824load_attr.29', '$840call.31', '$const834.30.3']
    $850build_tuple.32 = build_tuple(items=[Var($802call.26, rp_trees.py:811), Var($840call.31, rp_trees.py:811)]) ['$802call.26', '$840call.31', '$850build_tuple.32']
    $856call.33 = call $752load_method.21($850build_tuple.32, func=$752load_method.21, args=[Var($850build_tuple.32, rp_trees.py:811)], kws=(), vararg=None, varkwarg=None, target=None) ['$752load_method.21', '$850build_tuple.32', '$856call.33']
    $870load_method.36 = getattr(value=point_indices, attr=append) ['$870load_method.36', 'point_indices']
    $898call.38 = call $870load_method.36(indices, func=$870load_method.36, args=[Var(indices, rp_trees.py:749)], kws=(), vararg=None, varkwarg=None, target=None) ['$870load_method.36', '$898call.38', 'indices']
    jump 910                                 []
label 910:
    $const910.0.0 = const(NoneType, None)    ['$const910.0.0']
    $912return_value.1 = cast(value=$const910.0.0) ['$912return_value.1', '$const910.0.0']
    return $912return_value.1                ['$912return_value.1']
label 0:
    l = arg(0, name=l)                       ['l']
    $4load_global.0 = global(_list_length: <intrinsic _list_length>) ['$4load_global.0']
    $22call.3 = call $4load_global.0(l, func=$4load_global.0, args=[Var(l, listobject.py:412)], kws=(), vararg=None, varkwarg=None, target=None) ['$22call.3', '$4load_global.0', 'l']
    $32return_value.4 = cast(value=$22call.3) ['$22call.3', '$32return_value.4']
    return $32return_value.4                 ['$32return_value.4']
label 0:
    l = arg(0, name=l)                       ['l']
    item = arg(1, name=item)                 ['item']
    $6load_global.0 = global(_cast: <intrinsic _cast>) ['$6load_global.0']
    $20load_deref.3 = freevar(itemty: array(float32, 1d, C)) ['$20load_deref.3']
    casteditem = call $6load_global.0(item, $20load_deref.3, func=$6load_global.0, args=[Var(item, listobject.py:604), Var($20load_deref.3, listobject.py:605)], kws=(), vararg=None, varkwarg=None, target=None) ['$20load_deref.3', '$6load_global.0', 'casteditem', 'item']
    $38load_global.5 = global(_list_append: <intrinsic _list_append>) ['$38load_global.5']
    status = call $38load_global.5(l, casteditem, func=$38load_global.5, args=[Var(l, listobject.py:604), Var(casteditem, listobject.py:605)], kws=(), vararg=None, varkwarg=None, target=None) ['$38load_global.5', 'casteditem', 'l', 'status']
    $72load_global.11 = global(ListStatus: <enum 'ListStatus'>) ['$72load_global.11']
    $84load_attr.12 = getattr(value=$72load_global.11, attr=LIST_OK) ['$72load_global.11', '$84load_attr.12']
    $94compare_op.13 = status == $84load_attr.12 ['$84load_attr.12', '$94compare_op.13', 'status']
    bool100 = global(bool: <class 'bool'>)   ['bool100']
    $100pred = call bool100($94compare_op.13, func=bool100, args=(Var($94compare_op.13, listobject.py:607),), kws=(), vararg=None, varkwarg=None, target=None) ['$100pred', '$94compare_op.13', 'bool100']
    branch $100pred, 102, 106                ['$100pred']
label 102:
    $const102.0.0 = const(NoneType, None)    ['$const102.0.0']
    $104return_value.1 = cast(value=$const102.0.0) ['$104return_value.1', '$const102.0.0']
    return $104return_value.1                ['$104return_value.1']
label 106:
    $108load_global.1 = global(ListStatus: <enum 'ListStatus'>) ['$108load_global.1']
    $120load_attr.2 = getattr(value=$108load_global.1, attr=LIST_ERR_IMMUTABLE) ['$108load_global.1', '$120load_attr.2']
    $130compare_op.3 = status == $120load_attr.2 ['$120load_attr.2', '$130compare_op.3', 'status']
    bool136 = global(bool: <class 'bool'>)   ['bool136']
    $136pred = call bool136($130compare_op.3, func=bool136, args=(Var($130compare_op.3, listobject.py:609),), kws=(), vararg=None, varkwarg=None, target=None) ['$130compare_op.3', '$136pred', 'bool136']
    branch $136pred, 138, 168                ['$136pred']
label 138:
    $138load_global.0 = global(ValueError: <class 'ValueError'>) ['$138load_global.0']
    $const150.2.1 = const(str, list is immutable) ['$const150.2.1']
    $156call.3 = call $138load_global.0($const150.2.1, func=$138load_global.0, args=[Var($const150.2.1, listobject.py:610)], kws=(), vararg=None, varkwarg=None, target=None) ['$138load_global.0', '$156call.3', '$const150.2.1']
    raise $156call.3                         ['$156call.3']
label 168:
    $170load_global.1 = global(ListStatus: <enum 'ListStatus'>) ['$170load_global.1']
    $182load_attr.2 = getattr(value=$170load_global.1, attr=LIST_ERR_NO_MEMORY) ['$170load_global.1', '$182load_attr.2']
    $192compare_op.3 = status == $182load_attr.2 ['$182load_attr.2', '$192compare_op.3', 'status']
    bool198 = global(bool: <class 'bool'>)   ['bool198']
    $198pred = call bool198($192compare_op.3, func=bool198, args=(Var($192compare_op.3, listobject.py:611),), kws=(), vararg=None, varkwarg=None, target=None) ['$192compare_op.3', '$198pred', 'bool198']
    branch $198pred, 200, 230                ['$198pred']
label 200:
    $200load_global.0 = global(MemoryError: <class 'MemoryError'>) ['$200load_global.0']
    $const212.2.2 = const(str, Unable to allocate memory to append item) ['$const212.2.2']
    $218call.3 = call $200load_global.0($const212.2.2, func=$200load_global.0, args=[Var($const212.2.2, listobject.py:612)], kws=(), vararg=None, varkwarg=None, target=None) ['$200load_global.0', '$218call.3', '$const212.2.2']
    raise $218call.3                         ['$218call.3']
label 230:
    $230load_global.0 = global(RuntimeError: <class 'RuntimeError'>) ['$230load_global.0']
    $const242.2.3 = const(str, list.append failed unexpectedly) ['$const242.2.3']
    $248call.3 = call $230load_global.0($const242.2.3, func=$230load_global.0, args=[Var($const242.2.3, listobject.py:614)], kws=(), vararg=None, varkwarg=None, target=None) ['$230load_global.0', '$248call.3', '$const242.2.3']
    raise $248call.3                         ['$248call.3']
label 0:
    l = arg(0, name=l)                       ['l']
    item = arg(1, name=item)                 ['item']
    $6load_global.0 = global(_cast: <intrinsic _cast>) ['$6load_global.0']
    $20load_deref.3 = freevar(itemty: float64) ['$20load_deref.3']
    casteditem = call $6load_global.0(item, $20load_deref.3, func=$6load_global.0, args=[Var(item, listobject.py:604), Var($20load_deref.3, listobject.py:605)], kws=(), vararg=None, varkwarg=None, target=None) ['$20load_deref.3', '$6load_global.0', 'casteditem', 'item']
    $38load_global.5 = global(_list_append: <intrinsic _list_append>) ['$38load_global.5']
    status = call $38load_global.5(l, casteditem, func=$38load_global.5, args=[Var(l, listobject.py:604), Var(casteditem, listobject.py:605)], kws=(), vararg=None, varkwarg=None, target=None) ['$38load_global.5', 'casteditem', 'l', 'status']
    $72load_global.11 = global(ListStatus: <enum 'ListStatus'>) ['$72load_global.11']
    $84load_attr.12 = getattr(value=$72load_global.11, attr=LIST_OK) ['$72load_global.11', '$84load_attr.12']
    $94compare_op.13 = status == $84load_attr.12 ['$84load_attr.12', '$94compare_op.13', 'status']
    bool100 = global(bool: <class 'bool'>)   ['bool100']
    $100pred = call bool100($94compare_op.13, func=bool100, args=(Var($94compare_op.13, listobject.py:607),), kws=(), vararg=None, varkwarg=None, target=None) ['$100pred', '$94compare_op.13', 'bool100']
    branch $100pred, 102, 106                ['$100pred']
label 102:
    $const102.0.0 = const(NoneType, None)    ['$const102.0.0']
    $104return_value.1 = cast(value=$const102.0.0) ['$104return_value.1', '$const102.0.0']
    return $104return_value.1                ['$104return_value.1']
label 106:
    $108load_global.1 = global(ListStatus: <enum 'ListStatus'>) ['$108load_global.1']
    $120load_attr.2 = getattr(value=$108load_global.1, attr=LIST_ERR_IMMUTABLE) ['$108load_global.1', '$120load_attr.2']
    $130compare_op.3 = status == $120load_attr.2 ['$120load_attr.2', '$130compare_op.3', 'status']
    bool136 = global(bool: <class 'bool'>)   ['bool136']
    $136pred = call bool136($130compare_op.3, func=bool136, args=(Var($130compare_op.3, listobject.py:609),), kws=(), vararg=None, varkwarg=None, target=None) ['$130compare_op.3', '$136pred', 'bool136']
    branch $136pred, 138, 168                ['$136pred']
label 138:
    $138load_global.0 = global(ValueError: <class 'ValueError'>) ['$138load_global.0']
    $const150.2.1 = const(str, list is immutable) ['$const150.2.1']
    $156call.3 = call $138load_global.0($const150.2.1, func=$138load_global.0, args=[Var($const150.2.1, listobject.py:610)], kws=(), vararg=None, varkwarg=None, target=None) ['$138load_global.0', '$156call.3', '$const150.2.1']
    raise $156call.3                         ['$156call.3']
label 168:
    $170load_global.1 = global(ListStatus: <enum 'ListStatus'>) ['$170load_global.1']
    $182load_attr.2 = getattr(value=$170load_global.1, attr=LIST_ERR_NO_MEMORY) ['$170load_global.1', '$182load_attr.2']
    $192compare_op.3 = status == $182load_attr.2 ['$182load_attr.2', '$192compare_op.3', 'status']
    bool198 = global(bool: <class 'bool'>)   ['bool198']
    $198pred = call bool198($192compare_op.3, func=bool198, args=(Var($192compare_op.3, listobject.py:611),), kws=(), vararg=None, varkwarg=None, target=None) ['$192compare_op.3', '$198pred', 'bool198']
    branch $198pred, 200, 230                ['$198pred']
label 200:
    $200load_global.0 = global(MemoryError: <class 'MemoryError'>) ['$200load_global.0']
    $const212.2.2 = const(str, Unable to allocate memory to append item) ['$const212.2.2']
    $218call.3 = call $200load_global.0($const212.2.2, func=$200load_global.0, args=[Var($const212.2.2, listobject.py:612)], kws=(), vararg=None, varkwarg=None, target=None) ['$200load_global.0', '$218call.3', '$const212.2.2']
    raise $218call.3                         ['$218call.3']
label 230:
    $230load_global.0 = global(RuntimeError: <class 'RuntimeError'>) ['$230load_global.0']
    $const242.2.3 = const(str, list.append failed unexpectedly) ['$const242.2.3']
    $248call.3 = call $230load_global.0($const242.2.3, func=$230load_global.0, args=[Var($const242.2.3, listobject.py:614)], kws=(), vararg=None, varkwarg=None, target=None) ['$230load_global.0', '$248call.3', '$const242.2.3']
    raise $248call.3                         ['$248call.3']
label 0:
    object = arg(0, name=object)             ['object']
    dtype = arg(1, name=dtype)               ['dtype']
    $4load_global.0 = global(np_array: <intrinsic np_array>) ['$4load_global.0']
    $24call.4 = call $4load_global.0(object, dtype, func=$4load_global.0, args=[Var(object, arrayobj.py:5590), Var(dtype, arrayobj.py:5590)], kws=(), vararg=None, varkwarg=None, target=None) ['$24call.4', '$4load_global.0', 'dtype', 'object']
    $34return_value.5 = cast(value=$24call.4) ['$24call.4', '$34return_value.5']
    return $34return_value.5                 ['$34return_value.5']
label 0:
    l = arg(0, name=l)                       ['l']
    item = arg(1, name=item)                 ['item']
    $6load_global.0 = global(_cast: <intrinsic _cast>) ['$6load_global.0']
    $20load_deref.3 = freevar(itemty: array(int32, 1d, C)) ['$20load_deref.3']
    casteditem = call $6load_global.0(item, $20load_deref.3, func=$6load_global.0, args=[Var(item, listobject.py:604), Var($20load_deref.3, listobject.py:605)], kws=(), vararg=None, varkwarg=None, target=None) ['$20load_deref.3', '$6load_global.0', 'casteditem', 'item']
    $38load_global.5 = global(_list_append: <intrinsic _list_append>) ['$38load_global.5']
    status = call $38load_global.5(l, casteditem, func=$38load_global.5, args=[Var(l, listobject.py:604), Var(casteditem, listobject.py:605)], kws=(), vararg=None, varkwarg=None, target=None) ['$38load_global.5', 'casteditem', 'l', 'status']
    $72load_global.11 = global(ListStatus: <enum 'ListStatus'>) ['$72load_global.11']
    $84load_attr.12 = getattr(value=$72load_global.11, attr=LIST_OK) ['$72load_global.11', '$84load_attr.12']
    $94compare_op.13 = status == $84load_attr.12 ['$84load_attr.12', '$94compare_op.13', 'status']
    bool100 = global(bool: <class 'bool'>)   ['bool100']
    $100pred = call bool100($94compare_op.13, func=bool100, args=(Var($94compare_op.13, listobject.py:607),), kws=(), vararg=None, varkwarg=None, target=None) ['$100pred', '$94compare_op.13', 'bool100']
    branch $100pred, 102, 106                ['$100pred']
label 102:
    $const102.0.0 = const(NoneType, None)    ['$const102.0.0']
    $104return_value.1 = cast(value=$const102.0.0) ['$104return_value.1', '$const102.0.0']
    return $104return_value.1                ['$104return_value.1']
label 106:
    $108load_global.1 = global(ListStatus: <enum 'ListStatus'>) ['$108load_global.1']
    $120load_attr.2 = getattr(value=$108load_global.1, attr=LIST_ERR_IMMUTABLE) ['$108load_global.1', '$120load_attr.2']
    $130compare_op.3 = status == $120load_attr.2 ['$120load_attr.2', '$130compare_op.3', 'status']
    bool136 = global(bool: <class 'bool'>)   ['bool136']
    $136pred = call bool136($130compare_op.3, func=bool136, args=(Var($130compare_op.3, listobject.py:609),), kws=(), vararg=None, varkwarg=None, target=None) ['$130compare_op.3', '$136pred', 'bool136']
    branch $136pred, 138, 168                ['$136pred']
label 138:
    $138load_global.0 = global(ValueError: <class 'ValueError'>) ['$138load_global.0']
    $const150.2.1 = const(str, list is immutable) ['$const150.2.1']
    $156call.3 = call $138load_global.0($const150.2.1, func=$138load_global.0, args=[Var($const150.2.1, listobject.py:610)], kws=(), vararg=None, varkwarg=None, target=None) ['$138load_global.0', '$156call.3', '$const150.2.1']
    raise $156call.3                         ['$156call.3']
label 168:
    $170load_global.1 = global(ListStatus: <enum 'ListStatus'>) ['$170load_global.1']
    $182load_attr.2 = getattr(value=$170load_global.1, attr=LIST_ERR_NO_MEMORY) ['$170load_global.1', '$182load_attr.2']
    $192compare_op.3 = status == $182load_attr.2 ['$182load_attr.2', '$192compare_op.3', 'status']
    bool198 = global(bool: <class 'bool'>)   ['bool198']
    $198pred = call bool198($192compare_op.3, func=bool198, args=(Var($192compare_op.3, listobject.py:611),), kws=(), vararg=None, varkwarg=None, target=None) ['$192compare_op.3', '$198pred', 'bool198']
    branch $198pred, 200, 230                ['$198pred']
label 200:
    $200load_global.0 = global(MemoryError: <class 'MemoryError'>) ['$200load_global.0']
    $const212.2.2 = const(str, Unable to allocate memory to append item) ['$const212.2.2']
    $218call.3 = call $200load_global.0($const212.2.2, func=$200load_global.0, args=[Var($const212.2.2, listobject.py:612)], kws=(), vararg=None, varkwarg=None, target=None) ['$200load_global.0', '$218call.3', '$const212.2.2']
    raise $218call.3                         ['$218call.3']
label 230:
    $230load_global.0 = global(RuntimeError: <class 'RuntimeError'>) ['$230load_global.0']
    $const242.2.3 = const(str, list.append failed unexpectedly) ['$const242.2.3']
    $248call.3 = call $230load_global.0($const242.2.3, func=$230load_global.0, args=[Var($const242.2.3, listobject.py:614)], kws=(), vararg=None, varkwarg=None, target=None) ['$230load_global.0', '$248call.3', '$const242.2.3']
    raise $248call.3                         ['$248call.3']
label 0:
    object = arg(0, name=object)             ['object']
    dtype = arg(1, name=dtype)               ['dtype']
    $4load_global.0 = global(np_array: <intrinsic np_array>) ['$4load_global.0']
    $24call.4 = call $4load_global.0(object, dtype, func=$4load_global.0, args=[Var(object, arrayobj.py:5590), Var(dtype, arrayobj.py:5590)], kws=(), vararg=None, varkwarg=None, target=None) ['$24call.4', '$4load_global.0', 'dtype', 'object']
    $34return_value.5 = cast(value=$24call.4) ['$24call.4', '$34return_value.5']
    return $34return_value.5                 ['$34return_value.5']
label 0:
    l = arg(0, name=l)                       ['l']
    item = arg(1, name=item)                 ['item']
    $6load_global.0 = global(_cast: <intrinsic _cast>) ['$6load_global.0']
    $20load_deref.3 = freevar(itemty: UniTuple(int32 x 2)) ['$20load_deref.3']
    casteditem = call $6load_global.0(item, $20load_deref.3, func=$6load_global.0, args=[Var(item, listobject.py:604), Var($20load_deref.3, listobject.py:605)], kws=(), vararg=None, varkwarg=None, target=None) ['$20load_deref.3', '$6load_global.0', 'casteditem', 'item']
    $38load_global.5 = global(_list_append: <intrinsic _list_append>) ['$38load_global.5']
    status = call $38load_global.5(l, casteditem, func=$38load_global.5, args=[Var(l, listobject.py:604), Var(casteditem, listobject.py:605)], kws=(), vararg=None, varkwarg=None, target=None) ['$38load_global.5', 'casteditem', 'l', 'status']
    $72load_global.11 = global(ListStatus: <enum 'ListStatus'>) ['$72load_global.11']
    $84load_attr.12 = getattr(value=$72load_global.11, attr=LIST_OK) ['$72load_global.11', '$84load_attr.12']
    $94compare_op.13 = status == $84load_attr.12 ['$84load_attr.12', '$94compare_op.13', 'status']
    bool100 = global(bool: <class 'bool'>)   ['bool100']
    $100pred = call bool100($94compare_op.13, func=bool100, args=(Var($94compare_op.13, listobject.py:607),), kws=(), vararg=None, varkwarg=None, target=None) ['$100pred', '$94compare_op.13', 'bool100']
    branch $100pred, 102, 106                ['$100pred']
label 102:
    $const102.0.0 = const(NoneType, None)    ['$const102.0.0']
    $104return_value.1 = cast(value=$const102.0.0) ['$104return_value.1', '$const102.0.0']
    return $104return_value.1                ['$104return_value.1']
label 106:
    $108load_global.1 = global(ListStatus: <enum 'ListStatus'>) ['$108load_global.1']
    $120load_attr.2 = getattr(value=$108load_global.1, attr=LIST_ERR_IMMUTABLE) ['$108load_global.1', '$120load_attr.2']
    $130compare_op.3 = status == $120load_attr.2 ['$120load_attr.2', '$130compare_op.3', 'status']
    bool136 = global(bool: <class 'bool'>)   ['bool136']
    $136pred = call bool136($130compare_op.3, func=bool136, args=(Var($130compare_op.3, listobject.py:609),), kws=(), vararg=None, varkwarg=None, target=None) ['$130compare_op.3', '$136pred', 'bool136']
    branch $136pred, 138, 168                ['$136pred']
label 138:
    $138load_global.0 = global(ValueError: <class 'ValueError'>) ['$138load_global.0']
    $const150.2.1 = const(str, list is immutable) ['$const150.2.1']
    $156call.3 = call $138load_global.0($const150.2.1, func=$138load_global.0, args=[Var($const150.2.1, listobject.py:610)], kws=(), vararg=None, varkwarg=None, target=None) ['$138load_global.0', '$156call.3', '$const150.2.1']
    raise $156call.3                         ['$156call.3']
label 168:
    $170load_global.1 = global(ListStatus: <enum 'ListStatus'>) ['$170load_global.1']
    $182load_attr.2 = getattr(value=$170load_global.1, attr=LIST_ERR_NO_MEMORY) ['$170load_global.1', '$182load_attr.2']
    $192compare_op.3 = status == $182load_attr.2 ['$182load_attr.2', '$192compare_op.3', 'status']
    bool198 = global(bool: <class 'bool'>)   ['bool198']
    $198pred = call bool198($192compare_op.3, func=bool198, args=(Var($192compare_op.3, listobject.py:611),), kws=(), vararg=None, varkwarg=None, target=None) ['$192compare_op.3', '$198pred', 'bool198']
    branch $198pred, 200, 230                ['$198pred']
label 200:
    $200load_global.0 = global(MemoryError: <class 'MemoryError'>) ['$200load_global.0']
    $const212.2.2 = const(str, Unable to allocate memory to append item) ['$const212.2.2']
    $218call.3 = call $200load_global.0($const212.2.2, func=$200load_global.0, args=[Var($const212.2.2, listobject.py:612)], kws=(), vararg=None, varkwarg=None, target=None) ['$200load_global.0', '$218call.3', '$const212.2.2']
    raise $218call.3                         ['$218call.3']
label 230:
    $230load_global.0 = global(RuntimeError: <class 'RuntimeError'>) ['$230load_global.0']
    $const242.2.3 = const(str, list.append failed unexpectedly) ['$const242.2.3']
    $248call.3 = call $230load_global.0($const242.2.3, func=$230load_global.0, args=[Var($const242.2.3, listobject.py:614)], kws=(), vararg=None, varkwarg=None, target=None) ['$230load_global.0', '$248call.3', '$const242.2.3']
    raise $248call.3                         ['$248call.3']
label 0:
    shape = arg(0, name=shape)               ['shape']
    dtype = arg(1, name=dtype)               ['dtype']
    $6load_global.0 = global(numpy_empty_nd: <intrinsic numpy_empty_nd>) ['$6load_global.0']
    $22load_deref.4 = freevar(retty: array(float32, 1d, C)) ['$22load_deref.4']
    $28call.5 = call $6load_global.0(shape, dtype, $22load_deref.4, func=$6load_global.0, args=[Var(shape, arrayobj.py:4454), Var(dtype, arrayobj.py:4454), Var($22load_deref.4, arrayobj.py:4455)], kws=(), vararg=None, varkwarg=None, target=None) ['$22load_deref.4', '$28call.5', '$6load_global.0', 'dtype', 'shape']
    $38return_value.6 = cast(value=$28call.5) ['$28call.5', '$38return_value.6']
    return $38return_value.6                 ['$38return_value.6']
label 0:
    shape = arg(0, name=shape)               ['shape']
    dtype = arg(1, name=dtype)               ['dtype']
    $6load_global.0 = global(numpy_empty_nd: <intrinsic numpy_empty_nd>) ['$6load_global.0']
    $22load_deref.4 = freevar(retty: array(int32, 1d, C)) ['$22load_deref.4']
    $28call.5 = call $6load_global.0(shape, dtype, $22load_deref.4, func=$6load_global.0, args=[Var(shape, arrayobj.py:4454), Var(dtype, arrayobj.py:4454), Var($22load_deref.4, arrayobj.py:4455)], kws=(), vararg=None, varkwarg=None, target=None) ['$22load_deref.4', '$28call.5', '$6load_global.0', 'dtype', 'shape']
    $38return_value.6 = cast(value=$28call.5) ['$28call.5', '$38return_value.6']
    return $38return_value.6                 ['$38return_value.6']
label 0:
    data = arg(0, name=data)                 ['data']
    indices = arg(1, name=indices)           ['indices']
    hyperplanes = arg(2, name=hyperplanes)   ['hyperplanes']
    offsets = arg(3, name=offsets)           ['offsets']
    children = arg(4, name=children)         ['children']
    point_indices = arg(5, name=point_indices) ['point_indices']
    rng_state = arg(6, name=rng_state)       ['rng_state']
    leaf_size = arg(7, name=leaf_size)       ['leaf_size']
    max_depth = arg(8, name=max_depth)       ['max_depth']
    $6load_attr.1 = getattr(value=indices, attr=shape) ['$6load_attr.1', 'indices']
    $const16.2.1 = const(int, 0)             ['$const16.2.1']
    $18binary_subscr.3 = getitem(value=$6load_attr.1, index=$const16.2.1, fn=<built-in function getitem>) ['$18binary_subscr.3', '$6load_attr.1', '$const16.2.1']
    $30compare_op.5 = $18binary_subscr.3 > leaf_size ['$18binary_subscr.3', '$30compare_op.5', 'leaf_size']
    bool38 = global(bool: <class 'bool'>)    ['bool38']
    $38pred = call bool38($30compare_op.5, func=bool38, args=(Var($30compare_op.5, rp_trees.py:700),), kws=(), vararg=None, varkwarg=None, target=None) ['$30compare_op.5', '$38pred', 'bool38']
    branch $38pred, 40, 582                  ['$38pred']
label 40:
    $const42.1.1 = const(int, 0)             ['$const42.1.1']
    $44compare_op.2 = max_depth > $const42.1.1 ['$44compare_op.2', '$const42.1.1', 'max_depth']
    bool52 = global(bool: <class 'bool'>)    ['bool52']
    $52pred = call bool52($44compare_op.2, func=bool52, args=(Var($44compare_op.2, rp_trees.py:700),), kws=(), vararg=None, varkwarg=None, target=None) ['$44compare_op.2', '$52pred', 'bool52']
    branch $52pred, 54, 582                  ['$52pred']
label 54:
    $54load_global.0 = global(euclidean_random_projection_split: CPUDispatcher(<function euclidean_random_projection_split at 0x7fe1361d6ac0>)) ['$54load_global.0']
    $76call.5 = call $54load_global.0(data, indices, rng_state, func=$54load_global.0, args=[Var(data, rp_trees.py:685), Var(indices, rp_trees.py:685), Var(rng_state, rp_trees.py:685)], kws=(), vararg=None, varkwarg=None, target=None) ['$54load_global.0', '$76call.5', 'data', 'indices', 'rng_state']
    $86unpack_sequence.10 = exhaust_iter(value=$76call.5, count=4) ['$76call.5', '$86unpack_sequence.10']
    $86unpack_sequence.6 = static_getitem(value=$86unpack_sequence.10, index=0, index_var=None, fn=<built-in function getitem>) ['$86unpack_sequence.10', '$86unpack_sequence.6']
    $86unpack_sequence.7 = static_getitem(value=$86unpack_sequence.10, index=1, index_var=None, fn=<built-in function getitem>) ['$86unpack_sequence.10', '$86unpack_sequence.7']
    $86unpack_sequence.8 = static_getitem(value=$86unpack_sequence.10, index=2, index_var=None, fn=<built-in function getitem>) ['$86unpack_sequence.10', '$86unpack_sequence.8']
    $86unpack_sequence.9 = static_getitem(value=$86unpack_sequence.10, index=3, index_var=None, fn=<built-in function getitem>) ['$86unpack_sequence.10', '$86unpack_sequence.9']
    left_indices = $86unpack_sequence.6      ['$86unpack_sequence.6', 'left_indices']
    right_indices = $86unpack_sequence.7     ['$86unpack_sequence.7', 'right_indices']
    hyperplane = $86unpack_sequence.8        ['$86unpack_sequence.8', 'hyperplane']
    offset = $86unpack_sequence.9            ['$86unpack_sequence.9', 'offset']
    $98load_global.11 = global(make_euclidean_tree: CPUDispatcher(<function make_euclidean_tree at 0x7fe136407f60>)) ['$98load_global.11']
    $const128.22.2 = const(int, 1)           ['$const128.22.2']
    $binop_sub130.23 = max_depth - $const128.22.2 ['$binop_sub130.23', '$const128.22.2', 'max_depth']
    $138call.24 = call $98load_global.11(data, left_indices, hyperplanes, offsets, children, point_indices, rng_state, leaf_size, $binop_sub130.23, func=$98load_global.11, args=[Var(data, rp_trees.py:685), Var(left_indices, rp_trees.py:702), Var(hyperplanes, rp_trees.py:685), Var(offsets, rp_trees.py:685), Var(children, rp_trees.py:685), Var(point_indices, rp_trees.py:685), Var(rng_state, rp_trees.py:685), Var(leaf_size, rp_trees.py:685), Var($binop_sub130.23, rp_trees.py:717)], kws=(), vararg=None, varkwarg=None, target=None) ['$138call.24', '$98load_global.11', '$binop_sub130.23', 'children', 'data', 'hyperplanes', 'leaf_size', 'left_indices', 'offsets', 'point_indices', 'rng_state']
    $150load_global.25 = global(len: <built-in function len>) ['$150load_global.25']
    $168call.28 = call $150load_global.25(point_indices, func=$150load_global.25, args=[Var(point_indices, rp_trees.py:685)], kws=(), vararg=None, varkwarg=None, target=None) ['$150load_global.25', '$168call.28', 'point_indices']
    $const178.29.2 = const(int, 1)           ['$const178.29.2']
    left_node_num = $168call.28 - $const178.29.2 ['$168call.28', '$const178.29.2', 'left_node_num']
    $186load_global.31 = global(make_euclidean_tree: CPUDispatcher(<function make_euclidean_tree at 0x7fe136407f60>)) ['$186load_global.31']
    $const216.42.2 = const(int, 1)           ['$const216.42.2']
    $binop_sub218.43 = max_depth - $const216.42.2 ['$binop_sub218.43', '$const216.42.2', 'max_depth']
    $226call.44 = call $186load_global.31(data, right_indices, hyperplanes, offsets, children, point_indices, rng_state, leaf_size, $binop_sub218.43, func=$186load_global.31, args=[Var(data, rp_trees.py:685), Var(right_indices, rp_trees.py:703), Var(hyperplanes, rp_trees.py:685), Var(offsets, rp_trees.py:685), Var(children, rp_trees.py:685), Var(point_indices, rp_trees.py:685), Var(rng_state, rp_trees.py:685), Var(leaf_size, rp_trees.py:685), Var($binop_sub218.43, rp_trees.py:731)], kws=(), vararg=None, varkwarg=None, target=None) ['$186load_global.31', '$226call.44', '$binop_sub218.43', 'children', 'data', 'hyperplanes', 'leaf_size', 'offsets', 'point_indices', 'right_indices', 'rng_state']
    $238load_global.45 = global(len: <built-in function len>) ['$238load_global.45']
    $256call.48 = call $238load_global.45(point_indices, func=$238load_global.45, args=[Var(point_indices, rp_trees.py:685)], kws=(), vararg=None, varkwarg=None, target=None) ['$238load_global.45', '$256call.48', 'point_indices']
    $const266.49.2 = const(int, 1)           ['$const266.49.2']
    right_node_num = $256call.48 - $const266.49.2 ['$256call.48', '$const266.49.2', 'right_node_num']
    $276load_method.53 = getattr(value=hyperplanes, attr=append) ['$276load_method.53', 'hyperplanes']
    $304call.55 = call $276load_method.53(hyperplane, func=$276load_method.53, args=[Var(hyperplane, rp_trees.py:704)], kws=(), vararg=None, varkwarg=None, target=None) ['$276load_method.53', '$304call.55', 'hyperplane']
    $318load_method.58 = getattr(value=offsets, attr=append) ['$318load_method.58', 'offsets']
    $346call.60 = call $318load_method.58(offset, func=$318load_method.58, args=[Var(offset, rp_trees.py:705)], kws=(), vararg=None, varkwarg=None, target=None) ['$318load_method.58', '$346call.60', 'offset']
    $360load_method.63 = getattr(value=children, attr=append) ['$360load_method.63', 'children']
    $382load_global.64 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$382load_global.64']
    $394load_attr.66 = getattr(value=$382load_global.64, attr=int32) ['$382load_global.64', '$394load_attr.66']
    $410call.68 = call $394load_attr.66(left_node_num, func=$394load_attr.66, args=[Var(left_node_num, rp_trees.py:720)], kws=(), vararg=None, varkwarg=None, target=None) ['$394load_attr.66', '$410call.68', 'left_node_num']
    $420load_global.69 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$420load_global.69']
    $432load_attr.71 = getattr(value=$420load_global.69, attr=int32) ['$420load_global.69', '$432load_attr.71']
    $448call.73 = call $432load_attr.71(right_node_num, func=$432load_attr.71, args=[Var(right_node_num, rp_trees.py:734)], kws=(), vararg=None, varkwarg=None, target=None) ['$432load_attr.71', '$448call.73', 'right_node_num']
    $458build_tuple.74 = build_tuple(items=[Var($410call.68, rp_trees.py:738), Var($448call.73, rp_trees.py:738)]) ['$410call.68', '$448call.73', '$458build_tuple.74']
    $464call.75 = call $360load_method.63($458build_tuple.74, func=$360load_method.63, args=[Var($458build_tuple.74, rp_trees.py:738)], kws=(), vararg=None, varkwarg=None, target=None) ['$360load_method.63', '$458build_tuple.74', '$464call.75']
    $478load_method.78 = getattr(value=point_indices, attr=append) ['$478load_method.78', 'point_indices']
    $500load_global.79 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$500load_global.79']
    $512load_attr.81 = getattr(value=$500load_global.79, attr=array) ['$500load_global.79', '$512load_attr.81']
    $const522.82.3 = const(int, -1)          ['$const522.82.3']
    $524build_list.83 = build_list(items=[Var($const522.82.3, rp_trees.py:739)]) ['$524build_list.83', '$const522.82.3']
    $526load_global.84 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$526load_global.84']
    $538load_attr.85 = getattr(value=$526load_global.84, attr=int32) ['$526load_global.84', '$538load_attr.85']
    $554call.86 = call $512load_attr.81($524build_list.83, func=$512load_attr.81, args=[Var($524build_list.83, rp_trees.py:739)], kws=[('dtype', Var($538load_attr.85, rp_trees.py:739))], vararg=None, varkwarg=None, target=None) ['$512load_attr.81', '$524build_list.83', '$538load_attr.85', '$554call.86']
    $568call.87 = call $478load_method.78($554call.86, func=$478load_method.78, args=[Var($554call.86, rp_trees.py:739)], kws=(), vararg=None, varkwarg=None, target=None) ['$478load_method.78', '$554call.86', '$568call.87']
    jump 910                                 []
label 582:
    $584load_method.2 = getattr(value=hyperplanes, attr=append) ['$584load_method.2', 'hyperplanes']
    $606load_global.3 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$606load_global.3']
    $618load_attr.5 = getattr(value=$606load_global.3, attr=array) ['$606load_global.3', '$618load_attr.5']
    $const628.6.5 = const(float, -1.0)       ['$const628.6.5']
    $630build_list.7 = build_list(items=[Var($const628.6.5, rp_trees.py:741)]) ['$630build_list.7', '$const628.6.5']
    $632load_global.8 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$632load_global.8']
    $644load_attr.9 = getattr(value=$632load_global.8, attr=float32) ['$632load_global.8', '$644load_attr.9']
    $660call.10 = call $618load_attr.5($630build_list.7, func=$618load_attr.5, args=[Var($630build_list.7, rp_trees.py:741)], kws=[('dtype', Var($644load_attr.9, rp_trees.py:741))], vararg=None, varkwarg=None, target=None) ['$618load_attr.5', '$630build_list.7', '$644load_attr.9', '$660call.10']
    $674call.11 = call $584load_method.2($660call.10, func=$584load_method.2, args=[Var($660call.10, rp_trees.py:741)], kws=(), vararg=None, varkwarg=None, target=None) ['$584load_method.2', '$660call.10', '$674call.11']
    $688load_method.14 = getattr(value=offsets, attr=append) ['$688load_method.14', 'offsets']
    $710load_global.15 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$710load_global.15']
    $722load_attr.16 = getattr(value=$710load_global.15, attr=inf) ['$710load_global.15', '$722load_attr.16']
    $732unary_negative.17 = unary(fn=<built-in function neg>, value=$722load_attr.16) ['$722load_attr.16', '$732unary_negative.17']
    $738call.18 = call $688load_method.14($732unary_negative.17, func=$688load_method.14, args=[Var($732unary_negative.17, rp_trees.py:742)], kws=(), vararg=None, varkwarg=None, target=None) ['$688load_method.14', '$732unary_negative.17', '$738call.18']
    $752load_method.21 = getattr(value=children, attr=append) ['$752load_method.21', 'children']
    $774load_global.22 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$774load_global.22']
    $786load_attr.24 = getattr(value=$774load_global.22, attr=int32) ['$774load_global.22', '$786load_attr.24']
    $const796.25.3 = const(int, -1)          ['$const796.25.3']
    $802call.26 = call $786load_attr.24($const796.25.3, func=$786load_attr.24, args=[Var($const796.25.3, rp_trees.py:743)], kws=(), vararg=None, varkwarg=None, target=None) ['$786load_attr.24', '$802call.26', '$const796.25.3']
    $812load_global.27 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$812load_global.27']
    $824load_attr.29 = getattr(value=$812load_global.27, attr=int32) ['$812load_global.27', '$824load_attr.29']
    $const834.30.3 = const(int, -1)          ['$const834.30.3']
    $840call.31 = call $824load_attr.29($const834.30.3, func=$824load_attr.29, args=[Var($const834.30.3, rp_trees.py:743)], kws=(), vararg=None, varkwarg=None, target=None) ['$824load_attr.29', '$840call.31', '$const834.30.3']
    $850build_tuple.32 = build_tuple(items=[Var($802call.26, rp_trees.py:743), Var($840call.31, rp_trees.py:743)]) ['$802call.26', '$840call.31', '$850build_tuple.32']
    $856call.33 = call $752load_method.21($850build_tuple.32, func=$752load_method.21, args=[Var($850build_tuple.32, rp_trees.py:743)], kws=(), vararg=None, varkwarg=None, target=None) ['$752load_method.21', '$850build_tuple.32', '$856call.33']
    $870load_method.36 = getattr(value=point_indices, attr=append) ['$870load_method.36', 'point_indices']
    $898call.38 = call $870load_method.36(indices, func=$870load_method.36, args=[Var(indices, rp_trees.py:685)], kws=(), vararg=None, varkwarg=None, target=None) ['$870load_method.36', '$898call.38', 'indices']
    jump 910                                 []
label 910:
    $const910.0.0 = const(NoneType, None)    ['$const910.0.0']
    $912return_value.1 = cast(value=$const910.0.0) ['$912return_value.1', '$const910.0.0']
    return $912return_value.1                ['$912return_value.1']
label 0:
    cls = arg(0, name=cls)                   ['cls']
    item_type = arg(1, name=item_type)       ['item_type']
    allocated = arg(2, name=allocated)       ['allocated']
    $4load_global.0 = global(listobject: <module 'numba.typed.listobject' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/typed/listobject.py'>) ['$4load_global.0']
    $16load_attr.2 = getattr(value=$4load_global.0, attr=new_list) ['$16load_attr.2', '$4load_global.0']
    $36call.5 = call $16load_attr.2(item_type, func=$16load_attr.2, args=[Var(item_type, typedlist.py:453)], kws=[('allocated', Var(allocated, typedlist.py:453))], vararg=None, varkwarg=None, target=None) ['$16load_attr.2', '$36call.5', 'allocated', 'item_type']
    $46return_value.6 = cast(value=$36call.5) ['$36call.5', '$46return_value.6']
    return $46return_value.6                 ['$46return_value.6']
label 0:
    cls = arg(0, name=cls)                   ['cls']
    item_type = arg(1, name=item_type)       ['item_type']
    allocated = arg(2, name=allocated)       ['allocated']
    $4load_global.0 = global(listobject: <module 'numba.typed.listobject' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/typed/listobject.py'>) ['$4load_global.0']
    $16load_attr.2 = getattr(value=$4load_global.0, attr=new_list) ['$16load_attr.2', '$4load_global.0']
    $36call.5 = call $16load_attr.2(item_type, func=$16load_attr.2, args=[Var(item_type, typedlist.py:453)], kws=[('allocated', Var(allocated, typedlist.py:453))], vararg=None, varkwarg=None, target=None) ['$16load_attr.2', '$36call.5', 'allocated', 'item_type']
    $46return_value.6 = cast(value=$36call.5) ['$36call.5', '$46return_value.6']
    return $46return_value.6                 ['$46return_value.6']
label 0:
    cls = arg(0, name=cls)                   ['cls']
    item_type = arg(1, name=item_type)       ['item_type']
    allocated = arg(2, name=allocated)       ['allocated']
    $4load_global.0 = global(listobject: <module 'numba.typed.listobject' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/typed/listobject.py'>) ['$4load_global.0']
    $16load_attr.2 = getattr(value=$4load_global.0, attr=new_list) ['$16load_attr.2', '$4load_global.0']
    $36call.5 = call $16load_attr.2(item_type, func=$16load_attr.2, args=[Var(item_type, typedlist.py:453)], kws=[('allocated', Var(allocated, typedlist.py:453))], vararg=None, varkwarg=None, target=None) ['$16load_attr.2', '$36call.5', 'allocated', 'item_type']
    $46return_value.6 = cast(value=$36call.5) ['$36call.5', '$46return_value.6']
    return $46return_value.6                 ['$46return_value.6']
label 0:
    cls = arg(0, name=cls)                   ['cls']
    item_type = arg(1, name=item_type)       ['item_type']
    allocated = arg(2, name=allocated)       ['allocated']
    $4load_global.0 = global(listobject: <module 'numba.typed.listobject' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/typed/listobject.py'>) ['$4load_global.0']
    $16load_attr.2 = getattr(value=$4load_global.0, attr=new_list) ['$16load_attr.2', '$4load_global.0']
    $36call.5 = call $16load_attr.2(item_type, func=$16load_attr.2, args=[Var(item_type, typedlist.py:453)], kws=[('allocated', Var(allocated, typedlist.py:453))], vararg=None, varkwarg=None, target=None) ['$16load_attr.2', '$36call.5', 'allocated', 'item_type']
    $46return_value.6 = cast(value=$36call.5) ['$36call.5', '$46return_value.6']
    return $46return_value.6                 ['$46return_value.6']
label 0:
    tree = arg(0, name=tree)                 ['tree']
    max_leaf_size = arg(1, name=max_leaf_size) ['max_leaf_size']
    n_leaves = const(int, 0)                 ['n_leaves']
    $8load_global.1 = global(range: <class 'range'>) ['$8load_global.1']
    $20load_global.3 = global(len: <built-in function len>) ['$20load_global.3']
    $34load_attr.6 = getattr(value=tree, attr=children) ['$34load_attr.6', 'tree']
    $48call.7 = call $20load_global.3($34load_attr.6, func=$20load_global.3, args=[Var($34load_attr.6, rp_trees.py:1411)], kws=(), vararg=None, varkwarg=None, target=None) ['$20load_global.3', '$34load_attr.6', '$48call.7']
    $62call.8 = call $8load_global.1($48call.7, func=$8load_global.1, args=[Var($48call.7, rp_trees.py:1411)], kws=(), vararg=None, varkwarg=None, target=None) ['$48call.7', '$62call.8', '$8load_global.1']
    $72get_iter.9 = getiter(value=$62call.8) ['$62call.8', '$72get_iter.9']
    $phi74.0 = $72get_iter.9                 ['$72get_iter.9', '$phi74.0']
    jump 74                                  []
label 74:
    $74for_iter.1 = iternext(value=$phi74.0) ['$74for_iter.1', '$phi74.0']
    $74for_iter.2 = pair_first(value=$74for_iter.1) ['$74for_iter.1', '$74for_iter.2']
    $74for_iter.3 = pair_second(value=$74for_iter.1) ['$74for_iter.1', '$74for_iter.3']
    $phi76.1 = $74for_iter.2                 ['$74for_iter.2', '$phi76.1']
    branch $74for_iter.3, 76, 182            ['$74for_iter.3']
label 76:
    i = $phi76.1                             ['$phi76.1', 'i']
    $80load_attr.3 = getattr(value=tree, attr=children) ['$80load_attr.3', 'tree']
    $92binary_subscr.5 = getitem(value=$80load_attr.3, index=i, fn=<built-in function getitem>) ['$80load_attr.3', '$92binary_subscr.5', 'i']
    $const102.6.1 = const(int, 0)            ['$const102.6.1']
    $104binary_subscr.7 = getitem(value=$92binary_subscr.5, index=$const102.6.1, fn=<built-in function getitem>) ['$104binary_subscr.7', '$92binary_subscr.5', '$const102.6.1']
    $const114.8.2 = const(int, -1)           ['$const114.8.2']
    $116compare_op.9 = $104binary_subscr.7 == $const114.8.2 ['$104binary_subscr.7', '$116compare_op.9', '$const114.8.2']
    bool122 = global(bool: <class 'bool'>)   ['bool122']
    $122pred = call bool122($116compare_op.9, func=bool122, args=(Var($116compare_op.9, rp_trees.py:1412),), kws=(), vararg=None, varkwarg=None, target=None) ['$116compare_op.9', '$122pred', 'bool122']
    branch $122pred, 124, 180                ['$122pred']
label 124:
    $126load_attr.2 = getattr(value=tree, attr=children) ['$126load_attr.2', 'tree']
    $138binary_subscr.4 = getitem(value=$126load_attr.2, index=i, fn=<built-in function getitem>) ['$126load_attr.2', '$138binary_subscr.4', 'i']
    $const148.5.3 = const(int, 1)            ['$const148.5.3']
    $150binary_subscr.6 = getitem(value=$138binary_subscr.4, index=$const148.5.3, fn=<built-in function getitem>) ['$138binary_subscr.4', '$150binary_subscr.6', '$const148.5.3']
    $const160.7.2 = const(int, -1)           ['$const160.7.2']
    $162compare_op.8 = $150binary_subscr.6 == $const160.7.2 ['$150binary_subscr.6', '$162compare_op.8', '$const160.7.2']
    bool168 = global(bool: <class 'bool'>)   ['bool168']
    $168pred = call bool168($162compare_op.8, func=bool168, args=(Var($162compare_op.8, rp_trees.py:1412),), kws=(), vararg=None, varkwarg=None, target=None) ['$162compare_op.8', '$168pred', 'bool168']
    branch $168pred, 170, 180                ['$168pred']
label 170:
    $const172.2.3 = const(int, 1)            ['$const172.2.3']
    $binop_iadd174.3 = inplace_binop(fn=<built-in function iadd>, immutable_fn=<built-in function add>, lhs=n_leaves, rhs=$const172.2.3, static_lhs=Undefined, static_rhs=Undefined) ['$binop_iadd174.3', '$const172.2.3', 'n_leaves']
    n_leaves = $binop_iadd174.3              ['$binop_iadd174.3', 'n_leaves']
    jump 180                                 []
label 180:
    jump 74                                  []
label 182:
    $182load_global.0 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$182load_global.0']
    $194load_attr.2 = getattr(value=$182load_global.0, attr=full) ['$182load_global.0', '$194load_attr.2']
    $208build_tuple.5 = build_tuple(items=[Var(n_leaves, rp_trees.py:1410), Var(max_leaf_size, rp_trees.py:1408)]) ['$208build_tuple.5', 'max_leaf_size', 'n_leaves']
    $const210.6.2 = const(int, -1)           ['$const210.6.2']
    $212load_global.7 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$212load_global.7']
    $224load_attr.8 = getattr(value=$212load_global.7, attr=int32) ['$212load_global.7', '$224load_attr.8']
    result = call $194load_attr.2($208build_tuple.5, $const210.6.2, func=$194load_attr.2, args=[Var($208build_tuple.5, rp_trees.py:1415), Var($const210.6.2, rp_trees.py:1415)], kws=[('dtype', Var($224load_attr.8, rp_trees.py:1415))], vararg=None, varkwarg=None, target=None) ['$194load_attr.2', '$208build_tuple.5', '$224load_attr.8', '$const210.6.2', 'result']
    leaf_index = const(int, 0)               ['leaf_index']
    $256load_global.11 = global(range: <class 'range'>) ['$256load_global.11']
    $268load_global.13 = global(len: <built-in function len>) ['$268load_global.13']
    $282load_attr.16 = getattr(value=tree, attr=indices) ['$282load_attr.16', 'tree']
    $296call.17 = call $268load_global.13($282load_attr.16, func=$268load_global.13, args=[Var($282load_attr.16, rp_trees.py:1417)], kws=(), vararg=None, varkwarg=None, target=None) ['$268load_global.13', '$282load_attr.16', '$296call.17']
    $310call.18 = call $256load_global.11($296call.17, func=$256load_global.11, args=[Var($296call.17, rp_trees.py:1417)], kws=(), vararg=None, varkwarg=None, target=None) ['$256load_global.11', '$296call.17', '$310call.18']
    $320get_iter.19 = getiter(value=$310call.18) ['$310call.18', '$320get_iter.19']
    $phi322.0 = $320get_iter.19              ['$320get_iter.19', '$phi322.0']
    jump 322                                 []
label 322:
    $322for_iter.1 = iternext(value=$phi322.0) ['$322for_iter.1', '$phi322.0']
    $322for_iter.2 = pair_first(value=$322for_iter.1) ['$322for_iter.1', '$322for_iter.2']
    $322for_iter.3 = pair_second(value=$322for_iter.1) ['$322for_iter.1', '$322for_iter.3']
    $phi324.1 = $322for_iter.2               ['$322for_iter.2', '$phi324.1']
    branch $322for_iter.3, 324, 518          ['$322for_iter.3']
label 324:
    i = $phi324.1                            ['$phi324.1', 'i']
    $328load_attr.3 = getattr(value=tree, attr=children) ['$328load_attr.3', 'tree']
    $340binary_subscr.5 = getitem(value=$328load_attr.3, index=i, fn=<built-in function getitem>) ['$328load_attr.3', '$340binary_subscr.5', 'i']
    $const350.6.1 = const(int, 0)            ['$const350.6.1']
    $352binary_subscr.7 = getitem(value=$340binary_subscr.5, index=$const350.6.1, fn=<built-in function getitem>) ['$340binary_subscr.5', '$352binary_subscr.7', '$const350.6.1']
    $const362.8.2 = const(int, -1)           ['$const362.8.2']
    $364compare_op.9 = $352binary_subscr.7 == $const362.8.2 ['$352binary_subscr.7', '$364compare_op.9', '$const362.8.2']
    bool370 = global(bool: <class 'bool'>)   ['bool370']
    $370pred = call bool370($364compare_op.9, func=bool370, args=(Var($364compare_op.9, rp_trees.py:1418),), kws=(), vararg=None, varkwarg=None, target=None) ['$364compare_op.9', '$370pred', 'bool370']
    branch $370pred, 418, 372                ['$370pred']
label 372:
    $374load_attr.2 = getattr(value=tree, attr=children) ['$374load_attr.2', 'tree']
    $386binary_subscr.4 = getitem(value=$374load_attr.2, index=i, fn=<built-in function getitem>) ['$374load_attr.2', '$386binary_subscr.4', 'i']
    $const396.5.3 = const(int, 1)            ['$const396.5.3']
    $398binary_subscr.6 = getitem(value=$386binary_subscr.4, index=$const396.5.3, fn=<built-in function getitem>) ['$386binary_subscr.4', '$398binary_subscr.6', '$const396.5.3']
    $const408.7.2 = const(int, -1)           ['$const408.7.2']
    $410compare_op.8 = $398binary_subscr.6 == $const408.7.2 ['$398binary_subscr.6', '$410compare_op.8', '$const408.7.2']
    bool416 = global(bool: <class 'bool'>)   ['bool416']
    $416pred = call bool416($410compare_op.8, func=bool416, args=(Var($410compare_op.8, rp_trees.py:1418),), kws=(), vararg=None, varkwarg=None, target=None) ['$410compare_op.8', '$416pred', 'bool416']
    branch $416pred, 418, 516                ['$416pred']
label 418:
    $420load_attr.2 = getattr(value=tree, attr=indices) ['$420load_attr.2', 'tree']
    $432binary_subscr.4 = getitem(value=$420load_attr.2, index=i, fn=<built-in function getitem>) ['$420load_attr.2', '$432binary_subscr.4', 'i']
    $442load_attr.5 = getattr(value=$432binary_subscr.4, attr=shape) ['$432binary_subscr.4', '$442load_attr.5']
    $const452.6.1 = const(int, 0)            ['$const452.6.1']
    leaf_size = getitem(value=$442load_attr.5, index=$const452.6.1, fn=<built-in function getitem>) ['$442load_attr.5', '$const452.6.1', 'leaf_size']
    $468load_attr.9 = getattr(value=tree, attr=indices) ['$468load_attr.9', 'tree']
    $480binary_subscr.11 = getitem(value=$468load_attr.9, index=i, fn=<built-in function getitem>) ['$468load_attr.9', '$480binary_subscr.11', 'i']
    $const494.14.0 = const(NoneType, None)   ['$const494.14.0']
    $498build_slice.16 = global(slice: <class 'slice'>) ['$498build_slice.16']
    $498build_slice.17 = call $498build_slice.16($const494.14.0, leaf_size, func=$498build_slice.16, args=(Var($const494.14.0, rp_trees.py:1420), Var(leaf_size, rp_trees.py:1419)), kws=(), vararg=None, varkwarg=None, target=None) ['$498build_slice.16', '$498build_slice.17', '$const494.14.0', 'leaf_size']
    $500build_tuple.18 = build_tuple(items=[Var(leaf_index, rp_trees.py:1416), Var($498build_slice.17, rp_trees.py:1420)]) ['$498build_slice.17', '$500build_tuple.18', 'leaf_index']
    result[$500build_tuple.18] = $480binary_subscr.11 ['$480binary_subscr.11', '$500build_tuple.18', 'result']
    $const508.20.3 = const(int, 1)           ['$const508.20.3']
    $binop_iadd510.21 = inplace_binop(fn=<built-in function iadd>, immutable_fn=<built-in function add>, lhs=leaf_index, rhs=$const508.20.3, static_lhs=Undefined, static_rhs=Undefined) ['$binop_iadd510.21', '$const508.20.3', 'leaf_index']
    leaf_index = $binop_iadd510.21           ['$binop_iadd510.21', 'leaf_index']
    jump 516                                 []
label 516:
    jump 322                                 []
label 518:
    $520return_value.1 = cast(value=result)  ['$520return_value.1', 'result']
    return $520return_value.1                ['$520return_value.1']
label 0:
    l = arg(0, name=l)                       ['l']
    $4load_global.0 = global(_list_length: <intrinsic _list_length>) ['$4load_global.0']
    $22call.3 = call $4load_global.0(l, func=$4load_global.0, args=[Var(l, listobject.py:412)], kws=(), vararg=None, varkwarg=None, target=None) ['$22call.3', '$4load_global.0', 'l']
    $32return_value.4 = cast(value=$22call.3) ['$22call.3', '$32return_value.4']
    return $32return_value.4                 ['$32return_value.4']
label 0:
    l = arg(0, name=l)                       ['l']
    index = arg(1, name=index)               ['index']
    $6load_global.0 = global(_cast: <intrinsic _cast>) ['$6load_global.0']
    $20load_deref.3 = freevar(indexty: int64) ['$20load_deref.3']
    castedindex = call $6load_global.0(index, $20load_deref.3, func=$6load_global.0, args=[Var(index, listobject.py:771), Var($20load_deref.3, listobject.py:772)], kws=(), vararg=None, varkwarg=None, target=None) ['$20load_deref.3', '$6load_global.0', 'castedindex', 'index']
    $38load_global.5 = global(handle_index: <function handle_index at 0x7fe1371e54e0>) ['$38load_global.5']
    handledindex = call $38load_global.5(l, castedindex, func=$38load_global.5, args=[Var(l, listobject.py:771), Var(castedindex, listobject.py:772)], kws=(), vararg=None, varkwarg=None, target=None) ['$38load_global.5', 'castedindex', 'handledindex', 'l']
    $70load_global.10 = global(_list_getitem: <intrinsic impl>) ['$70load_global.10']
    $90call.14 = call $70load_global.10(l, handledindex, func=$70load_global.10, args=[Var(l, listobject.py:771), Var(handledindex, listobject.py:773)], kws=(), vararg=None, varkwarg=None, target=None) ['$70load_global.10', '$90call.14', 'handledindex', 'l']
    $100unpack_sequence.17 = exhaust_iter(value=$90call.14, count=2) ['$100unpack_sequence.17', '$90call.14']
    $100unpack_sequence.15 = static_getitem(value=$100unpack_sequence.17, index=0, index_var=None, fn=<built-in function getitem>) ['$100unpack_sequence.15', '$100unpack_sequence.17']
    $100unpack_sequence.16 = static_getitem(value=$100unpack_sequence.17, index=1, index_var=None, fn=<built-in function getitem>) ['$100unpack_sequence.16', '$100unpack_sequence.17']
    status = $100unpack_sequence.15          ['$100unpack_sequence.15', 'status']
    item = $100unpack_sequence.16            ['$100unpack_sequence.16', 'item']
    $110load_global.19 = global(ListStatus: <enum 'ListStatus'>) ['$110load_global.19']
    $122load_attr.20 = getattr(value=$110load_global.19, attr=LIST_OK) ['$110load_global.19', '$122load_attr.20']
    $132compare_op.21 = status == $122load_attr.20 ['$122load_attr.20', '$132compare_op.21', 'status']
    bool138 = global(bool: <class 'bool'>)   ['bool138']
    $138pred = call bool138($132compare_op.21, func=bool138, args=(Var($132compare_op.21, listobject.py:775),), kws=(), vararg=None, varkwarg=None, target=None) ['$132compare_op.21', '$138pred', 'bool138']
    branch $138pred, 140, 170                ['$138pred']
label 140:
    $140load_global.0 = global(_nonoptional: <intrinsic _nonoptional>) ['$140load_global.0']
    $158call.3 = call $140load_global.0(item, func=$140load_global.0, args=[Var(item, listobject.py:774)], kws=(), vararg=None, varkwarg=None, target=None) ['$140load_global.0', '$158call.3', 'item']
    $168return_value.4 = cast(value=$158call.3) ['$158call.3', '$168return_value.4']
    return $168return_value.4                ['$168return_value.4']
label 170:
    $170load_global.0 = global(AssertionError: <class 'AssertionError'>) ['$170load_global.0']
    $const182.2.1 = const(str, internal list error during getitem) ['$const182.2.1']
    $188call.3 = call $170load_global.0($const182.2.1, func=$170load_global.0, args=[Var($const182.2.1, listobject.py:778)], kws=(), vararg=None, varkwarg=None, target=None) ['$170load_global.0', '$188call.3', '$const182.2.1']
    raise $188call.3                         ['$188call.3']
label 0:
    l = arg(0, name=l)                       ['l']
    index = arg(1, name=index)               ['index']
    $4load_global.0 = global(fix_index: <intrinsic fix_index>) ['$4load_global.0']
    index.1 = call $4load_global.0(l, index, func=$4load_global.0, args=[Var(l, listobject.py:656), Var(index, listobject.py:656)], kws=(), vararg=None, varkwarg=None, target=None) ['$4load_global.0', 'index', 'index.1', 'l']
    $const38.6.1 = const(int, 0)             ['$const38.6.1']
    $40compare_op.7 = index.1 < $const38.6.1 ['$40compare_op.7', '$const38.6.1', 'index.1']
    bool46 = global(bool: <class 'bool'>)    ['bool46']
    $46pred = call bool46($40compare_op.7, func=bool46, args=(Var($40compare_op.7, listobject.py:666),), kws=(), vararg=None, varkwarg=None, target=None) ['$40compare_op.7', '$46pred', 'bool46']
    branch $46pred, 86, 48                   ['$46pred']
label 48:
    $50load_global.1 = global(len: <built-in function len>) ['$50load_global.1']
    $68call.4 = call $50load_global.1(l, func=$50load_global.1, args=[Var(l, listobject.py:656)], kws=(), vararg=None, varkwarg=None, target=None) ['$50load_global.1', '$68call.4', 'l']
    $78compare_op.5 = index.1 >= $68call.4   ['$68call.4', '$78compare_op.5', 'index.1']
    bool84 = global(bool: <class 'bool'>)    ['bool84']
    $84pred = call bool84($78compare_op.5, func=bool84, args=(Var($78compare_op.5, listobject.py:666),), kws=(), vararg=None, varkwarg=None, target=None) ['$78compare_op.5', '$84pred', 'bool84']
    branch $84pred, 86, 116                  ['$84pred']
label 86:
    $86load_global.0 = global(IndexError: <class 'IndexError'>) ['$86load_global.0']
    $const98.2.2 = const(str, list index out of range) ['$const98.2.2']
    $104call.3 = call $86load_global.0($const98.2.2, func=$86load_global.0, args=[Var($const98.2.2, listobject.py:667)], kws=(), vararg=None, varkwarg=None, target=None) ['$104call.3', '$86load_global.0', '$const98.2.2']
    raise $104call.3                         ['$104call.3']
label 116:
    $118return_value.1 = cast(value=index.1) ['$118return_value.1', 'index.1']
    return $118return_value.1                ['$118return_value.1']
label 0:
    l = arg(0, name=l)                       ['l']
    $4load_global.0 = global(_list_length: <intrinsic _list_length>) ['$4load_global.0']
    $22call.3 = call $4load_global.0(l, func=$4load_global.0, args=[Var(l, listobject.py:412)], kws=(), vararg=None, varkwarg=None, target=None) ['$22call.3', '$4load_global.0', 'l']
    $32return_value.4 = cast(value=$22call.3) ['$22call.3', '$32return_value.4']
    return $32return_value.4                 ['$32return_value.4']
label 0:
    shape = arg(0, name=shape)               ['shape']
    fill_value = arg(1, name=fill_value)     ['fill_value']
    dtype = arg(2, name=dtype)               ['dtype']
    $6load_global.0 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$6load_global.0']
    $18load_attr.2 = getattr(value=$6load_global.0, attr=empty) ['$18load_attr.2', '$6load_global.0']
    $30load_deref.4 = freevar(nb_dtype: int32) ['$30load_deref.4']
    arr = call $18load_attr.2(shape, $30load_deref.4, func=$18load_attr.2, args=[Var(shape, arrayobj.py:4563), Var($30load_deref.4, arrayobj.py:4564)], kws=(), vararg=None, varkwarg=None, target=None) ['$18load_attr.2', '$30load_deref.4', 'arr', 'shape']
    arr_flat = getattr(value=arr, attr=flat) ['arr', 'arr_flat']
    $62load_global.8 = global(range: <class 'range'>) ['$62load_global.8']
    $74load_global.10 = global(len: <built-in function len>) ['$74load_global.10']
    $92call.13 = call $74load_global.10(arr_flat, func=$74load_global.10, args=[Var(arr_flat, arrayobj.py:4565)], kws=(), vararg=None, varkwarg=None, target=None) ['$74load_global.10', '$92call.13', 'arr_flat']
    $106call.14 = call $62load_global.8($92call.13, func=$62load_global.8, args=[Var($92call.13, arrayobj.py:4566)], kws=(), vararg=None, varkwarg=None, target=None) ['$106call.14', '$62load_global.8', '$92call.13']
    $116get_iter.15 = getiter(value=$106call.14) ['$106call.14', '$116get_iter.15']
    $phi118.0 = $116get_iter.15              ['$116get_iter.15', '$phi118.0']
    jump 118                                 []
label 118:
    $118for_iter.1 = iternext(value=$phi118.0) ['$118for_iter.1', '$phi118.0']
    $118for_iter.2 = pair_first(value=$118for_iter.1) ['$118for_iter.1', '$118for_iter.2']
    $118for_iter.3 = pair_second(value=$118for_iter.1) ['$118for_iter.1', '$118for_iter.3']
    $phi120.1 = $118for_iter.2               ['$118for_iter.2', '$phi120.1']
    branch $118for_iter.3, 120, 134          ['$118for_iter.3']
label 120:
    idx = $phi120.1                          ['$phi120.1', 'idx']
    arr_flat[idx] = fill_value               ['arr_flat', 'fill_value', 'idx']
    jump 118                                 []
label 134:
    $136return_value.1 = cast(value=arr)     ['$136return_value.1', 'arr']
    return $136return_value.1                ['$136return_value.1']
label 0:
    l = arg(0, name=l)                       ['l']
    index = arg(1, name=index)               ['index']
    $6load_global.0 = global(_cast: <intrinsic _cast>) ['$6load_global.0']
    $20load_deref.3 = freevar(indexty: int64) ['$20load_deref.3']
    castedindex = call $6load_global.0(index, $20load_deref.3, func=$6load_global.0, args=[Var(index, listobject.py:771), Var($20load_deref.3, listobject.py:772)], kws=(), vararg=None, varkwarg=None, target=None) ['$20load_deref.3', '$6load_global.0', 'castedindex', 'index']
    $38load_global.5 = global(handle_index: <function handle_index at 0x7fe1371e54e0>) ['$38load_global.5']
    handledindex = call $38load_global.5(l, castedindex, func=$38load_global.5, args=[Var(l, listobject.py:771), Var(castedindex, listobject.py:772)], kws=(), vararg=None, varkwarg=None, target=None) ['$38load_global.5', 'castedindex', 'handledindex', 'l']
    $70load_global.10 = global(_list_getitem: <intrinsic impl>) ['$70load_global.10']
    $90call.14 = call $70load_global.10(l, handledindex, func=$70load_global.10, args=[Var(l, listobject.py:771), Var(handledindex, listobject.py:773)], kws=(), vararg=None, varkwarg=None, target=None) ['$70load_global.10', '$90call.14', 'handledindex', 'l']
    $100unpack_sequence.17 = exhaust_iter(value=$90call.14, count=2) ['$100unpack_sequence.17', '$90call.14']
    $100unpack_sequence.15 = static_getitem(value=$100unpack_sequence.17, index=0, index_var=None, fn=<built-in function getitem>) ['$100unpack_sequence.15', '$100unpack_sequence.17']
    $100unpack_sequence.16 = static_getitem(value=$100unpack_sequence.17, index=1, index_var=None, fn=<built-in function getitem>) ['$100unpack_sequence.16', '$100unpack_sequence.17']
    status = $100unpack_sequence.15          ['$100unpack_sequence.15', 'status']
    item = $100unpack_sequence.16            ['$100unpack_sequence.16', 'item']
    $110load_global.19 = global(ListStatus: <enum 'ListStatus'>) ['$110load_global.19']
    $122load_attr.20 = getattr(value=$110load_global.19, attr=LIST_OK) ['$110load_global.19', '$122load_attr.20']
    $132compare_op.21 = status == $122load_attr.20 ['$122load_attr.20', '$132compare_op.21', 'status']
    bool138 = global(bool: <class 'bool'>)   ['bool138']
    $138pred = call bool138($132compare_op.21, func=bool138, args=(Var($132compare_op.21, listobject.py:775),), kws=(), vararg=None, varkwarg=None, target=None) ['$132compare_op.21', '$138pred', 'bool138']
    branch $138pred, 140, 170                ['$138pred']
label 140:
    $140load_global.0 = global(_nonoptional: <intrinsic _nonoptional>) ['$140load_global.0']
    $158call.3 = call $140load_global.0(item, func=$140load_global.0, args=[Var(item, listobject.py:774)], kws=(), vararg=None, varkwarg=None, target=None) ['$140load_global.0', '$158call.3', 'item']
    $168return_value.4 = cast(value=$158call.3) ['$158call.3', '$168return_value.4']
    return $168return_value.4                ['$168return_value.4']
label 170:
    $170load_global.0 = global(AssertionError: <class 'AssertionError'>) ['$170load_global.0']
    $const182.2.1 = const(str, internal list error during getitem) ['$const182.2.1']
    $188call.3 = call $170load_global.0($const182.2.1, func=$170load_global.0, args=[Var($const182.2.1, listobject.py:778)], kws=(), vararg=None, varkwarg=None, target=None) ['$170load_global.0', '$188call.3', '$const182.2.1']
    raise $188call.3                         ['$188call.3']
label 0:
    l = arg(0, name=l)                       ['l']
    index = arg(1, name=index)               ['index']
    $4load_global.0 = global(fix_index: <intrinsic fix_index>) ['$4load_global.0']
    index.1 = call $4load_global.0(l, index, func=$4load_global.0, args=[Var(l, listobject.py:656), Var(index, listobject.py:656)], kws=(), vararg=None, varkwarg=None, target=None) ['$4load_global.0', 'index', 'index.1', 'l']
    $const38.6.1 = const(int, 0)             ['$const38.6.1']
    $40compare_op.7 = index.1 < $const38.6.1 ['$40compare_op.7', '$const38.6.1', 'index.1']
    bool46 = global(bool: <class 'bool'>)    ['bool46']
    $46pred = call bool46($40compare_op.7, func=bool46, args=(Var($40compare_op.7, listobject.py:666),), kws=(), vararg=None, varkwarg=None, target=None) ['$40compare_op.7', '$46pred', 'bool46']
    branch $46pred, 86, 48                   ['$46pred']
label 48:
    $50load_global.1 = global(len: <built-in function len>) ['$50load_global.1']
    $68call.4 = call $50load_global.1(l, func=$50load_global.1, args=[Var(l, listobject.py:656)], kws=(), vararg=None, varkwarg=None, target=None) ['$50load_global.1', '$68call.4', 'l']
    $78compare_op.5 = index.1 >= $68call.4   ['$68call.4', '$78compare_op.5', 'index.1']
    bool84 = global(bool: <class 'bool'>)    ['bool84']
    $84pred = call bool84($78compare_op.5, func=bool84, args=(Var($78compare_op.5, listobject.py:666),), kws=(), vararg=None, varkwarg=None, target=None) ['$78compare_op.5', '$84pred', 'bool84']
    branch $84pred, 86, 116                  ['$84pred']
label 86:
    $86load_global.0 = global(IndexError: <class 'IndexError'>) ['$86load_global.0']
    $const98.2.2 = const(str, list index out of range) ['$const98.2.2']
    $104call.3 = call $86load_global.0($const98.2.2, func=$86load_global.0, args=[Var($const98.2.2, listobject.py:667)], kws=(), vararg=None, varkwarg=None, target=None) ['$104call.3', '$86load_global.0', '$const98.2.2']
    raise $104call.3                         ['$104call.3']
label 116:
    $118return_value.1 = cast(value=index.1) ['$118return_value.1', 'index.1']
    return $118return_value.1                ['$118return_value.1']
label 0:
    l = arg(0, name=l)                       ['l']
    $4load_global.0 = global(_list_length: <intrinsic _list_length>) ['$4load_global.0']
    $22call.3 = call $4load_global.0(l, func=$4load_global.0, args=[Var(l, listobject.py:412)], kws=(), vararg=None, varkwarg=None, target=None) ['$22call.3', '$4load_global.0', 'l']
    $32return_value.4 = cast(value=$22call.3) ['$22call.3', '$32return_value.4']
    return $32return_value.4                 ['$32return_value.4']
label 0:
    src_shapes = arg(0, name=src_shapes)     ['src_shapes']
    index_shape = arg(1, name=index_shape)   ['index_shape']
    $4load_global.0 = global(len: <built-in function len>) ['$4load_global.0']
    $22call.3 = call $4load_global.0(src_shapes, func=$4load_global.0, args=[Var(src_shapes, arrayobj.py:1650)], kws=(), vararg=None, varkwarg=None, target=None) ['$22call.3', '$4load_global.0', 'src_shapes']
    $const32.4.1 = const(int, 1)             ['$const32.4.1']
    $34compare_op.5 = $22call.3 == $const32.4.1 ['$22call.3', '$34compare_op.5', '$const32.4.1']
    bool40 = global(bool: <class 'bool'>)    ['bool40']
    $40pred = call bool40($34compare_op.5, func=bool40, args=(Var($34compare_op.5, arrayobj.py:1651),), kws=(), vararg=None, varkwarg=None, target=None) ['$34compare_op.5', '$40pred', 'bool40']
    branch $40pred, 42, 68                   ['$40pred']
label 42:
    $const42.0.2 = const(str, ()             ['$const42.0.2']
    $const46.2.3 = const(int, 0)             ['$const46.2.3']
    $48binary_subscr.3 = getitem(value=src_shapes, index=$const46.2.3, fn=<built-in function getitem>) ['$48binary_subscr.3', '$const46.2.3', 'src_shapes']
    $58format_value.4 = global(str: <class 'str'>) ['$58format_value.4']
    $58format_value.5 = call $58format_value.4($48binary_subscr.3, func=$58format_value.4, args=(Var($48binary_subscr.3, arrayobj.py:1652),), kws=(), vararg=None, varkwarg=None, target=None) ['$48binary_subscr.3', '$58format_value.4', '$58format_value.5']
    $const60.6.4 = const(str, ,))            ['$const60.6.4']
    $62build_string.7 = $const42.0.2 + $58format_value.5 ['$58format_value.5', '$62build_string.7', '$const42.0.2']
    shape_str = $62build_string.7 + $const60.6.4 ['$62build_string.7', '$const60.6.4', 'shape_str']
    jump 138                                 []
label 68:
    $const68.0.2 = const(str, ()             ['$const68.0.2']
    $const70.1.5 = const(str, , )            ['$const70.1.5']
    $72load_method.3 = getattr(value=$const70.1.5, attr=join) ['$72load_method.3', '$const70.1.5']
    $const94.4.6 = const(code, <code object <listcomp> at 0x7fe1374323d0, file "/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/np/arrayobj.py", line 1654>) ['$const94.4.6']
    $96make_function.5 = make_function(name=None, code=<code object <listcomp> at 0x7fe1374323d0, file "/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/np/arrayobj.py", line 1654>, closure=None, defaults=None) ['$96make_function.5']
    $100get_iter.7 = getiter(value=src_shapes) ['$100get_iter.7', 'src_shapes']
    $106call.8 = call $96make_function.5($100get_iter.7, func=$96make_function.5, args=[Var($100get_iter.7, arrayobj.py:1654)], kws=(), vararg=None, varkwarg=None, target=None) ['$100get_iter.7', '$106call.8', '$96make_function.5']
    $120call.9 = call $72load_method.3($106call.8, func=$72load_method.3, args=[Var($106call.8, arrayobj.py:1654)], kws=(), vararg=None, varkwarg=None, target=None) ['$106call.8', '$120call.9', '$72load_method.3']
    $130format_value.10 = global(str: <class 'str'>) ['$130format_value.10']
    $130format_value.11 = call $130format_value.10($120call.9, func=$130format_value.10, args=(Var($120call.9, arrayobj.py:1654),), kws=(), vararg=None, varkwarg=None, target=None) ['$120call.9', '$130format_value.10', '$130format_value.11']
    $const132.12.7 = const(str, ))           ['$const132.12.7']
    $134build_string.13 = $const68.0.2 + $130format_value.11 ['$130format_value.11', '$134build_string.13', '$const68.0.2']
    shape_str = $134build_string.13 + $const132.12.7 ['$134build_string.13', '$const132.12.7', 'shape_str']
    jump 138                                 []
label 138:
    $138load_global.0 = global(len: <built-in function len>) ['$138load_global.0']
    $156call.3 = call $138load_global.0(index_shape, func=$138load_global.0, args=[Var(index_shape, arrayobj.py:1650)], kws=(), vararg=None, varkwarg=None, target=None) ['$138load_global.0', '$156call.3', 'index_shape']
    $const166.4.1 = const(int, 1)            ['$const166.4.1']
    $168compare_op.5 = $156call.3 == $const166.4.1 ['$156call.3', '$168compare_op.5', '$const166.4.1']
    bool174 = global(bool: <class 'bool'>)   ['bool174']
    $174pred = call bool174($168compare_op.5, func=bool174, args=(Var($168compare_op.5, arrayobj.py:1655),), kws=(), vararg=None, varkwarg=None, target=None) ['$168compare_op.5', '$174pred', 'bool174']
    branch $174pred, 176, 202                ['$174pred']
label 176:
    $const176.0.2 = const(str, ()            ['$const176.0.2']
    $const180.2.3 = const(int, 0)            ['$const180.2.3']
    $182binary_subscr.3 = getitem(value=index_shape, index=$const180.2.3, fn=<built-in function getitem>) ['$182binary_subscr.3', '$const180.2.3', 'index_shape']
    $192format_value.4 = global(str: <class 'str'>) ['$192format_value.4']
    $192format_value.5 = call $192format_value.4($182binary_subscr.3, func=$192format_value.4, args=(Var($182binary_subscr.3, arrayobj.py:1656),), kws=(), vararg=None, varkwarg=None, target=None) ['$182binary_subscr.3', '$192format_value.4', '$192format_value.5']
    $const194.6.4 = const(str, ,))           ['$const194.6.4']
    $196build_string.7 = $const176.0.2 + $192format_value.5 ['$192format_value.5', '$196build_string.7', '$const176.0.2']
    index_str = $196build_string.7 + $const194.6.4 ['$196build_string.7', '$const194.6.4', 'index_str']
    jump 272                                 []
label 202:
    $const202.0.2 = const(str, ()            ['$const202.0.2']
    $const204.1.5 = const(str, , )           ['$const204.1.5']
    $206load_method.3 = getattr(value=$const204.1.5, attr=join) ['$206load_method.3', '$const204.1.5']
    $const228.4.8 = const(code, <code object <listcomp> at 0x7fe1374324c0, file "/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/np/arrayobj.py", line 1658>) ['$const228.4.8']
    $230make_function.5 = make_function(name=None, code=<code object <listcomp> at 0x7fe1374324c0, file "/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/np/arrayobj.py", line 1658>, closure=None, defaults=None) ['$230make_function.5']
    $234get_iter.7 = getiter(value=index_shape) ['$234get_iter.7', 'index_shape']
    $240call.8 = call $230make_function.5($234get_iter.7, func=$230make_function.5, args=[Var($234get_iter.7, arrayobj.py:1658)], kws=(), vararg=None, varkwarg=None, target=None) ['$230make_function.5', '$234get_iter.7', '$240call.8']
    $254call.9 = call $206load_method.3($240call.8, func=$206load_method.3, args=[Var($240call.8, arrayobj.py:1658)], kws=(), vararg=None, varkwarg=None, target=None) ['$206load_method.3', '$240call.8', '$254call.9']
    $264format_value.10 = global(str: <class 'str'>) ['$264format_value.10']
    $264format_value.11 = call $264format_value.10($254call.9, func=$264format_value.10, args=(Var($254call.9, arrayobj.py:1658),), kws=(), vararg=None, varkwarg=None, target=None) ['$254call.9', '$264format_value.10', '$264format_value.11']
    $const266.12.7 = const(str, ))           ['$const266.12.7']
    $268build_string.13 = $const202.0.2 + $264format_value.11 ['$264format_value.11', '$268build_string.13', '$const202.0.2']
    index_str = $268build_string.13 + $const266.12.7 ['$268build_string.13', '$const266.12.7', 'index_str']
    jump 272                                 []
label 272:
    $const272.0.9 = const(str, cannot assign slice of shape ) ['$const272.0.9']
    $276format_value.2 = global(str: <class 'str'>) ['$276format_value.2']
    $276format_value.3 = call $276format_value.2(shape_str, func=$276format_value.2, args=(Var(shape_str, arrayobj.py:1652),), kws=(), vararg=None, varkwarg=None, target=None) ['$276format_value.2', '$276format_value.3', 'shape_str']
    $const278.4.10 = const(str,  from input of shape ) ['$const278.4.10']
    $282format_value.6 = global(str: <class 'str'>) ['$282format_value.6']
    $282format_value.7 = call $282format_value.6(index_str, func=$282format_value.6, args=(Var(index_str, arrayobj.py:1656),), kws=(), vararg=None, varkwarg=None, target=None) ['$282format_value.6', '$282format_value.7', 'index_str']
    $284build_string.8 = $const272.0.9 + $276format_value.3 ['$276format_value.3', '$284build_string.8', '$const272.0.9']
    $284build_string.9 = $284build_string.8 + $const278.4.10 ['$284build_string.8', '$284build_string.9', '$const278.4.10']
    msg = $284build_string.9 + $282format_value.7 ['$282format_value.7', '$284build_string.9', 'msg']
    $288load_global.11 = global(ValueError: <class 'ValueError'>) ['$288load_global.11']
    $306call.14 = call $288load_global.11(msg, func=$288load_global.11, args=[Var(msg, arrayobj.py:1659)], kws=(), vararg=None, varkwarg=None, target=None) ['$288load_global.11', '$306call.14', 'msg']
    raise $306call.14                        ['$306call.14']
label 0:
    implicit0 = arg(0, name=implicit0)       ['implicit0']
    $4build_list.0 = build_list(items=[])    ['$4build_list.0']
    $phi8.1 = implicit0                      ['$phi8.1', 'implicit0']
    $phi44.0 = $4build_list.0                ['$4build_list.0', '$phi44.0']
    $phi10.0 = $4build_list.0                ['$4build_list.0', '$phi10.0']
    jump 8                                   []
label 8:
    $8for_iter.2 = iternext(value=$phi8.1)   ['$8for_iter.2', '$phi8.1']
    $8for_iter.3 = pair_first(value=$8for_iter.2) ['$8for_iter.2', '$8for_iter.3']
    $8for_iter.4 = pair_second(value=$8for_iter.2) ['$8for_iter.2', '$8for_iter.4']
    $phi10.2 = $8for_iter.3                  ['$8for_iter.3', '$phi10.2']
    branch $8for_iter.4, 10, 44              ['$8for_iter.4']
label 10:
    x = $phi10.2                             ['$phi10.2', 'x']
    $12load_global.3 = global(str: <class 'str'>) ['$12load_global.3']
    $30call.6 = call $12load_global.3(x, func=$12load_global.3, args=[Var(x, arrayobj.py:1658)], kws=(), vararg=None, varkwarg=None, target=None) ['$12load_global.3', '$30call.6', 'x']
    $40list_append.7 = getattr(value=$phi10.0, attr=append) ['$40list_append.7', '$phi10.0']
    $40list_append.8 = call $40list_append.7($30call.6, func=$40list_append.7, args=(Var($30call.6, arrayobj.py:1658),), kws=(), vararg=None, varkwarg=None, target=None) ['$30call.6', '$40list_append.7', '$40list_append.8']
    jump 8                                   []
label 44:
    $44return_value.1 = cast(value=$phi44.0) ['$44return_value.1', '$phi44.0']
    return $44return_value.1                 ['$44return_value.1']
label 0:
    implicit0 = arg(0, name=implicit0)       ['implicit0']
    $4build_list.0 = build_list(items=[])    ['$4build_list.0']
    $phi8.1 = implicit0                      ['$phi8.1', 'implicit0']
    $phi44.0 = $4build_list.0                ['$4build_list.0', '$phi44.0']
    $phi10.0 = $4build_list.0                ['$4build_list.0', '$phi10.0']
    jump 8                                   []
label 8:
    $8for_iter.2 = iternext(value=$phi8.1)   ['$8for_iter.2', '$phi8.1']
    $8for_iter.3 = pair_first(value=$8for_iter.2) ['$8for_iter.2', '$8for_iter.3']
    $8for_iter.4 = pair_second(value=$8for_iter.2) ['$8for_iter.2', '$8for_iter.4']
    $phi10.2 = $8for_iter.3                  ['$8for_iter.3', '$phi10.2']
    branch $8for_iter.4, 10, 44              ['$8for_iter.4']
label 10:
    x = $phi10.2                             ['$phi10.2', 'x']
    $12load_global.3 = global(str: <class 'str'>) ['$12load_global.3']
    $30call.6 = call $12load_global.3(x, func=$12load_global.3, args=[Var(x, arrayobj.py:1654)], kws=(), vararg=None, varkwarg=None, target=None) ['$12load_global.3', '$30call.6', 'x']
    $40list_append.7 = getattr(value=$phi10.0, attr=append) ['$40list_append.7', '$phi10.0']
    $40list_append.8 = call $40list_append.7($30call.6, func=$40list_append.7, args=(Var($30call.6, arrayobj.py:1654),), kws=(), vararg=None, varkwarg=None, target=None) ['$30call.6', '$40list_append.7', '$40list_append.8']
    jump 8                                   []
label 44:
    $44return_value.1 = cast(value=$phi44.0) ['$44return_value.1', '$phi44.0']
    return $44return_value.1                 ['$44return_value.1']
label 0:
    object = arg(0, name=object)             ['object']
    attr = const(str, __str__)               ['attr']
    $8load_global.1 = global(hasattr: <built-in function hasattr>) ['$8load_global.1']
    $28call.5 = call $8load_global.1(object, attr, func=$8load_global.1, args=[Var(object, old_builtins.py:1019), Var(attr, old_builtins.py:1020)], kws=(), vararg=None, varkwarg=None, target=None) ['$28call.5', '$8load_global.1', 'attr', 'object']
    $const38.6.2 = const(bool, True)         ['$const38.6.2']
    $40compare_op.7 = $28call.5 == $const38.6.2 ['$28call.5', '$40compare_op.7', '$const38.6.2']
    bool46 = global(bool: <class 'bool'>)    ['bool46']
    $46pred = call bool46($40compare_op.7, func=bool46, args=(Var($40compare_op.7, old_builtins.py:1021),), kws=(), vararg=None, varkwarg=None, target=None) ['$40compare_op.7', '$46pred', 'bool46']
    branch $46pred, 48, 96                   ['$46pred']
label 48:
    $50load_global.1 = global(getattr: <built-in function getattr>) ['$50load_global.1']
    $70call.5 = call $50load_global.1(object, attr, func=$50load_global.1, args=[Var(object, old_builtins.py:1019), Var(attr, old_builtins.py:1020)], kws=(), vararg=None, varkwarg=None, target=None) ['$50load_global.1', '$70call.5', 'attr', 'object']
    $84call.6 = call $70call.5(func=$70call.5, args=[], kws=(), vararg=None, varkwarg=None, target=None) ['$70call.5', '$84call.6']
    $94return_value.7 = cast(value=$84call.6) ['$84call.6', '$94return_value.7']
    return $94return_value.7                 ['$94return_value.7']
label 96:
    $96load_global.0 = global(repr: <built-in function repr>) ['$96load_global.0']
    $114call.3 = call $96load_global.0(object, func=$96load_global.0, args=[Var(object, old_builtins.py:1019)], kws=(), vararg=None, varkwarg=None, target=None) ['$114call.3', '$96load_global.0', 'object']
    $124return_value.4 = cast(value=$114call.3) ['$114call.3', '$124return_value.4']
    return $124return_value.4                ['$124return_value.4']
label 0:
    obj = arg(0, name=obj)                   ['obj']
    name = arg(1, name=name)                 ['name']
    $4load_global.0 = global(resolve_hasattr: <intrinsic resolve_hasattr>) ['$4load_global.0']
    $24call.4 = call $4load_global.0(obj, name, func=$4load_global.0, args=[Var(obj, old_builtins.py:998), Var(name, old_builtins.py:998)], kws=(), vararg=None, varkwarg=None, target=None) ['$24call.4', '$4load_global.0', 'name', 'obj']
    $34return_value.5 = cast(value=$24call.4) ['$24call.4', '$34return_value.5']
    return $34return_value.5                 ['$34return_value.5']
label 0:
    obj = arg(0, name=obj)                   ['obj']
    name = arg(1, name=name)                 ['name']
    $4load_global.0 = global(resolve_hasattr: <intrinsic resolve_hasattr>) ['$4load_global.0']
    $24call.4 = call $4load_global.0(obj, name, func=$4load_global.0, args=[Var(obj, old_builtins.py:998), Var(name, old_builtins.py:998)], kws=(), vararg=None, varkwarg=None, target=None) ['$24call.4', '$4load_global.0', 'name', 'obj']
    $34return_value.5 = cast(value=$24call.4) ['$24call.4', '$34return_value.5']
    return $34return_value.5                 ['$34return_value.5']
label 0:
    obj = arg(0, name=obj)                   ['obj']
    name = arg(1, name=name)                 ['name']
    $4load_global.0 = global(resolve_getattr: <intrinsic resolve_getattr>) ['$4load_global.0']
    $20load_global.4 = global(_getattr_default: _getattr_default_type()) ['$20load_global.4']
    $36call.5 = call $4load_global.0(obj, name, $20load_global.4, func=$4load_global.0, args=[Var(obj, old_builtins.py:954), Var(name, old_builtins.py:954), Var($20load_global.4, old_builtins.py:955)], kws=(), vararg=None, varkwarg=None, target=None) ['$20load_global.4', '$36call.5', '$4load_global.0', 'name', 'obj']
    $46return_value.6 = cast(value=$36call.5) ['$36call.5', '$46return_value.6']
    return $46return_value.6                 ['$46return_value.6']
label 0:
    n = arg(0, name=n)                       ['n']
    flag = const(bool, False)                ['flag']
    $const12.2.2 = const(int, 0)             ['$const12.2.2']
    $14compare_op.3 = n < $const12.2.2       ['$14compare_op.3', '$const12.2.2', 'n']
    bool20 = global(bool: <class 'bool'>)    ['bool20']
    $20pred = call bool20($14compare_op.3, func=bool20, args=(Var($14compare_op.3, unicode.py:2574),), kws=(), vararg=None, varkwarg=None, target=None) ['$14compare_op.3', '$20pred', 'bool20']
    branch $20pred, 22, 32                   ['$20pred']
label 22:
    n = unary(fn=<built-in function neg>, value=n) ['n', 'n']
    flag = const(bool, True)                 ['flag']
    jump 32                                  []
label 32:
    $const34.1.2 = const(int, 0)             ['$const34.1.2']
    $36compare_op.2 = n == $const34.1.2      ['$36compare_op.2', '$const34.1.2', 'n']
    bool42 = global(bool: <class 'bool'>)    ['bool42']
    $42pred = call bool42($36compare_op.2, func=bool42, args=(Var($36compare_op.2, unicode.py:2577),), kws=(), vararg=None, varkwarg=None, target=None) ['$36compare_op.2', '$42pred', 'bool42']
    branch $42pred, 44, 48                   ['$42pred']
label 44:
    $const44.0.4 = const(str, 0)             ['$const44.0.4']
    $46return_value.1 = cast(value=$const44.0.4) ['$46return_value.1', '$const44.0.4']
    return $46return_value.1                 ['$46return_value.1']
label 48:
    $const50.1.5 = const(int, 1)             ['$const50.1.5']
    $binop_add52.2 = flag + $const50.1.5     ['$binop_add52.2', '$const50.1.5', 'flag']
    $56load_global.3 = global(int: <class 'int'>) ['$56load_global.3']
    $68load_global.5 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$68load_global.5']
    $80load_attr.7 = getattr(value=$68load_global.5, attr=floor) ['$68load_global.5', '$80load_attr.7']
    $90load_global.8 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$90load_global.8']
    $102load_attr.10 = getattr(value=$90load_global.8, attr=log10) ['$102load_attr.10', '$90load_global.8']
    $118call.12 = call $102load_attr.10(n, func=$102load_attr.10, args=[Var(n, unicode.py:2572)], kws=(), vararg=None, varkwarg=None, target=None) ['$102load_attr.10', '$118call.12', 'n']
    $132call.13 = call $80load_attr.7($118call.12, func=$80load_attr.7, args=[Var($118call.12, unicode.py:2579)], kws=(), vararg=None, varkwarg=None, target=None) ['$118call.12', '$132call.13', '$80load_attr.7']
    $146call.14 = call $56load_global.3($132call.13, func=$56load_global.3, args=[Var($132call.13, unicode.py:2579)], kws=(), vararg=None, varkwarg=None, target=None) ['$132call.13', '$146call.14', '$56load_global.3']
    length = $binop_add52.2 + $146call.14    ['$146call.14', '$binop_add52.2', 'length']
    kind = global(PY_UNICODE_1BYTE_KIND: 1)  ['kind']
    $176load_global.17 = global(_kind_to_byte_width: <function _kind_to_byte_width at 0x7fe1e54e5c60>) ['$176load_global.17']
    char_width = call $176load_global.17(kind, func=$176load_global.17, args=[Var(kind, unicode.py:2580)], kws=(), vararg=None, varkwarg=None, target=None) ['$176load_global.17', 'char_width', 'kind']
    $206load_global.21 = global(_malloc_string: <intrinsic _malloc_string>) ['$206load_global.21']
    $const224.26.3 = const(bool, True)       ['$const224.26.3']
    s = call $206load_global.21(kind, char_width, length, $const224.26.3, func=$206load_global.21, args=[Var(kind, unicode.py:2580), Var(char_width, unicode.py:2581), Var(length, unicode.py:2579), Var($const224.26.3, unicode.py:2582)], kws=(), vararg=None, varkwarg=None, target=None) ['$206load_global.21', '$const224.26.3', 'char_width', 'kind', 'length', 's']
    bool244 = global(bool: <class 'bool'>)   ['bool244']
    $244pred = call bool244(flag, func=bool244, args=(Var(flag, unicode.py:2573),), kws=(), vararg=None, varkwarg=None, target=None) ['$244pred', 'bool244', 'flag']
    branch $244pred, 246, 306                ['$244pred']
label 246:
    $246load_global.0 = global(_set_code_point: <function _set_code_point at 0x7fe1e54e5a80>) ['$246load_global.0']
    $const260.3.2 = const(int, 0)            ['$const260.3.2']
    $262load_global.4 = global(ord: <built-in function ord>) ['$262load_global.4']
    $const274.6.6 = const(str, -)            ['$const274.6.6']
    $280call.7 = call $262load_global.4($const274.6.6, func=$262load_global.4, args=[Var($const274.6.6, unicode.py:2584)], kws=(), vararg=None, varkwarg=None, target=None) ['$262load_global.4', '$280call.7', '$const274.6.6']
    $294call.8 = call $246load_global.0(s, $const260.3.2, $280call.7, func=$246load_global.0, args=[Var(s, unicode.py:2582), Var($const260.3.2, unicode.py:2584), Var($280call.7, unicode.py:2584)], kws=(), vararg=None, varkwarg=None, target=None) ['$246load_global.0', '$280call.7', '$294call.8', '$const260.3.2', 's']
    jump 306                                 []
label 306:
    $const308.1.5 = const(int, 1)            ['$const308.1.5']
    idx = length - $const308.1.5             ['$const308.1.5', 'idx', 'length']
    $const318.4.2 = const(int, 0)            ['$const318.4.2']
    $320compare_op.5 = n > $const318.4.2     ['$320compare_op.5', '$const318.4.2', 'n']
    bool326 = global(bool: <class 'bool'>)   ['bool326']
    $326pred = call bool326($320compare_op.5, func=bool326, args=(Var($320compare_op.5, unicode.py:2586),), kws=(), vararg=None, varkwarg=None, target=None) ['$320compare_op.5', '$326pred', 'bool326']
    branch $326pred, 328, 458                ['$326pred']
label 328:
    $328load_global.0 = global(divmod: <built-in function divmod>) ['$328load_global.0']
    $342load_deref.3 = freevar(ten: 10)      ['$342load_deref.3']
    $348call.4 = call $328load_global.0(n, $342load_deref.3, func=$328load_global.0, args=[Var(n, unicode.py:2572), Var($342load_deref.3, unicode.py:2587)], kws=(), vararg=None, varkwarg=None, target=None) ['$328load_global.0', '$342load_deref.3', '$348call.4', 'n']
    $358unpack_sequence.7 = exhaust_iter(value=$348call.4, count=2) ['$348call.4', '$358unpack_sequence.7']
    $358unpack_sequence.5 = static_getitem(value=$358unpack_sequence.7, index=0, index_var=None, fn=<built-in function getitem>) ['$358unpack_sequence.5', '$358unpack_sequence.7']
    $358unpack_sequence.6 = static_getitem(value=$358unpack_sequence.7, index=1, index_var=None, fn=<built-in function getitem>) ['$358unpack_sequence.6', '$358unpack_sequence.7']
    n = $358unpack_sequence.5                ['$358unpack_sequence.5', 'n']
    digit = $358unpack_sequence.6            ['$358unpack_sequence.6', 'digit']
    $366load_global.8 = global(ord: <built-in function ord>) ['$366load_global.8']
    $const378.10.4 = const(str, 0)           ['$const378.10.4']
    $384call.11 = call $366load_global.8($const378.10.4, func=$366load_global.8, args=[Var($const378.10.4, unicode.py:2588)], kws=(), vararg=None, varkwarg=None, target=None) ['$366load_global.8', '$384call.11', '$const378.10.4']
    c = $384call.11 + digit                  ['$384call.11', 'c', 'digit']
    $402load_global.14 = global(_set_code_point: <function _set_code_point at 0x7fe1e54e5a80>) ['$402load_global.14']
    $424call.19 = call $402load_global.14(s, idx, c, func=$402load_global.14, args=[Var(s, unicode.py:2582), Var(idx, unicode.py:2585), Var(c, unicode.py:2588)], kws=(), vararg=None, varkwarg=None, target=None) ['$402load_global.14', '$424call.19', 'c', 'idx', 's']
    $const438.21.5 = const(int, 1)           ['$const438.21.5']
    $binop_isub440.22 = inplace_binop(fn=<built-in function isub>, immutable_fn=<built-in function sub>, lhs=idx, rhs=$const438.21.5, static_lhs=Undefined, static_rhs=Undefined) ['$binop_isub440.22', '$const438.21.5', 'idx']
    idx = $binop_isub440.22                  ['$binop_isub440.22', 'idx']
    $const448.24.2 = const(int, 0)           ['$const448.24.2']
    $450compare_op.25 = n > $const448.24.2   ['$450compare_op.25', '$const448.24.2', 'n']
    bool456 = global(bool: <class 'bool'>)   ['bool456']
    $456pred = call bool456($450compare_op.25, func=bool456, args=(Var($450compare_op.25, unicode.py:2586),), kws=(), vararg=None, varkwarg=None, target=None) ['$450compare_op.25', '$456pred', 'bool456']
    branch $456pred, 328, 458                ['$456pred']
label 458:
    $460return_value.1 = cast(value=s)       ['$460return_value.1', 's']
    return $460return_value.1                ['$460return_value.1']
label 0:
    kind = arg(0, name=kind)                 ['kind']
    $6load_global.1 = global(PY_UNICODE_1BYTE_KIND: 1) ['$6load_global.1']
    $18compare_op.2 = kind == $6load_global.1 ['$18compare_op.2', '$6load_global.1', 'kind']
    bool24 = global(bool: <class 'bool'>)    ['bool24']
    $24pred = call bool24($18compare_op.2, func=bool24, args=(Var($18compare_op.2, unicode.py:413),), kws=(), vararg=None, varkwarg=None, target=None) ['$18compare_op.2', '$24pred', 'bool24']
    branch $24pred, 26, 30                   ['$24pred']
label 26:
    $const26.0.1 = const(int, 1)             ['$const26.0.1']
    $28return_value.1 = cast(value=$const26.0.1) ['$28return_value.1', '$const26.0.1']
    return $28return_value.1                 ['$28return_value.1']
label 30:
    $32load_global.1 = global(PY_UNICODE_2BYTE_KIND: 2) ['$32load_global.1']
    $44compare_op.2 = kind == $32load_global.1 ['$32load_global.1', '$44compare_op.2', 'kind']
    bool50 = global(bool: <class 'bool'>)    ['bool50']
    $50pred = call bool50($44compare_op.2, func=bool50, args=(Var($44compare_op.2, unicode.py:415),), kws=(), vararg=None, varkwarg=None, target=None) ['$44compare_op.2', '$50pred', 'bool50']
    branch $50pred, 52, 56                   ['$50pred']
label 52:
    $const52.0.2 = const(int, 2)             ['$const52.0.2']
    $54return_value.1 = cast(value=$const52.0.2) ['$54return_value.1', '$const52.0.2']
    return $54return_value.1                 ['$54return_value.1']
label 56:
    $58load_global.1 = global(PY_UNICODE_4BYTE_KIND: 4) ['$58load_global.1']
    $70compare_op.2 = kind == $58load_global.1 ['$58load_global.1', '$70compare_op.2', 'kind']
    bool76 = global(bool: <class 'bool'>)    ['bool76']
    $76pred = call bool76($70compare_op.2, func=bool76, args=(Var($70compare_op.2, unicode.py:417),), kws=(), vararg=None, varkwarg=None, target=None) ['$70compare_op.2', '$76pred', 'bool76']
    branch $76pred, 78, 82                   ['$76pred']
label 78:
    $const78.0.3 = const(int, 4)             ['$const78.0.3']
    $80return_value.1 = cast(value=$const78.0.3) ['$80return_value.1', '$const78.0.3']
    return $80return_value.1                 ['$80return_value.1']
label 82:
    $84load_global.1 = global(PY_UNICODE_WCHAR_KIND: 0) ['$84load_global.1']
    $96compare_op.2 = kind == $84load_global.1 ['$84load_global.1', '$96compare_op.2', 'kind']
    bool102 = global(bool: <class 'bool'>)   ['bool102']
    $102pred = call bool102($96compare_op.2, func=bool102, args=(Var($96compare_op.2, unicode.py:419),), kws=(), vararg=None, varkwarg=None, target=None) ['$102pred', '$96compare_op.2', 'bool102']
    branch $102pred, 104, 134                ['$102pred']
label 104:
    $104load_global.0 = global(AssertionError: <class 'AssertionError'>) ['$104load_global.0']
    $const116.2.4 = const(str, PY_UNICODE_WCHAR_KIND unsupported) ['$const116.2.4']
    $122call.3 = call $104load_global.0($const116.2.4, func=$104load_global.0, args=[Var($const116.2.4, unicode.py:420)], kws=(), vararg=None, varkwarg=None, target=None) ['$104load_global.0', '$122call.3', '$const116.2.4']
    raise $122call.3                         ['$122call.3']
label 134:
    $134load_global.0 = global(AssertionError: <class 'AssertionError'>) ['$134load_global.0']
    $const146.2.5 = const(str, Unexpected unicode encoding encountered) ['$const146.2.5']
    $152call.3 = call $134load_global.0($const146.2.5, func=$134load_global.0, args=[Var($const146.2.5, unicode.py:422)], kws=(), vararg=None, varkwarg=None, target=None) ['$134load_global.0', '$152call.3', '$const146.2.5']
    raise $152call.3                         ['$152call.3']
label 0:
    c = arg(0, name=c)                       ['c']
    $4load_global.0 = global(len: <built-in function len>) ['$4load_global.0']
    lc = call $4load_global.0(c, func=$4load_global.0, args=[Var(c, unicode.py:2510)], kws=(), vararg=None, varkwarg=None, target=None) ['$4load_global.0', 'c', 'lc']
    $const36.5.1 = const(int, 1)             ['$const36.5.1']
    $38compare_op.6 = lc != $const36.5.1     ['$38compare_op.6', '$const36.5.1', 'lc']
    bool44 = global(bool: <class 'bool'>)    ['bool44']
    $44pred = call bool44($38compare_op.6, func=bool44, args=(Var($38compare_op.6, unicode.py:2512),), kws=(), vararg=None, varkwarg=None, target=None) ['$38compare_op.6', '$44pred', 'bool44']
    branch $44pred, 46, 76                   ['$44pred']
label 46:
    $46load_global.0 = global(TypeError: <class 'TypeError'>) ['$46load_global.0']
    $const58.2.2 = const(str, ord() expected a character) ['$const58.2.2']
    $64call.3 = call $46load_global.0($const58.2.2, func=$46load_global.0, args=[Var($const58.2.2, unicode.py:2514)], kws=(), vararg=None, varkwarg=None, target=None) ['$46load_global.0', '$64call.3', '$const58.2.2']
    raise $64call.3                          ['$64call.3']
label 76:
    $76load_global.0 = global(_get_code_point: <function _get_code_point at 0x7fe1e54e5620>) ['$76load_global.0']
    $const90.3.3 = const(int, 0)             ['$const90.3.3']
    $96call.4 = call $76load_global.0(c, $const90.3.3, func=$76load_global.0, args=[Var(c, unicode.py:2510), Var($const90.3.3, unicode.py:2515)], kws=(), vararg=None, varkwarg=None, target=None) ['$76load_global.0', '$96call.4', '$const90.3.3', 'c']
    $106return_value.5 = cast(value=$96call.4) ['$106return_value.5', '$96call.4']
    return $106return_value.5                ['$106return_value.5']
label 0:
    s = arg(0, name=s)                       ['s']
    $6load_attr.1 = getattr(value=s, attr=_length) ['$6load_attr.1', 's']
    $16return_value.2 = cast(value=$6load_attr.1) ['$16return_value.2', '$6load_attr.1']
    return $16return_value.2                 ['$16return_value.2']
label 0:
    a = arg(0, name=a)                       ['a']
    i = arg(1, name=i)                       ['i']
    $6load_attr.1 = getattr(value=a, attr=_kind) ['$6load_attr.1', 'a']
    $16load_global.2 = global(PY_UNICODE_1BYTE_KIND: 1) ['$16load_global.2']
    $28compare_op.3 = $6load_attr.1 == $16load_global.2 ['$16load_global.2', '$28compare_op.3', '$6load_attr.1']
    bool34 = global(bool: <class 'bool'>)    ['bool34']
    $34pred = call bool34($28compare_op.3, func=bool34, args=(Var($28compare_op.3, unicode.py:292),), kws=(), vararg=None, varkwarg=None, target=None) ['$28compare_op.3', '$34pred', 'bool34']
    branch $34pred, 36, 78                   ['$34pred']
label 36:
    $36load_global.0 = global(deref_uint8: <intrinsic deref_uint8>) ['$36load_global.0']
    $50load_attr.3 = getattr(value=a, attr=_data) ['$50load_attr.3', 'a']
    $66call.5 = call $36load_global.0($50load_attr.3, i, func=$36load_global.0, args=[Var($50load_attr.3, unicode.py:293), Var(i, unicode.py:290)], kws=(), vararg=None, varkwarg=None, target=None) ['$36load_global.0', '$50load_attr.3', '$66call.5', 'i']
    $76return_value.6 = cast(value=$66call.5) ['$66call.5', '$76return_value.6']
    return $76return_value.6                 ['$76return_value.6']
label 78:
    $80load_attr.1 = getattr(value=a, attr=_kind) ['$80load_attr.1', 'a']
    $90load_global.2 = global(PY_UNICODE_2BYTE_KIND: 2) ['$90load_global.2']
    $102compare_op.3 = $80load_attr.1 == $90load_global.2 ['$102compare_op.3', '$80load_attr.1', '$90load_global.2']
    bool108 = global(bool: <class 'bool'>)   ['bool108']
    $108pred = call bool108($102compare_op.3, func=bool108, args=(Var($102compare_op.3, unicode.py:294),), kws=(), vararg=None, varkwarg=None, target=None) ['$102compare_op.3', '$108pred', 'bool108']
    branch $108pred, 110, 152                ['$108pred']
label 110:
    $110load_global.0 = global(deref_uint16: <intrinsic deref_uint16>) ['$110load_global.0']
    $124load_attr.3 = getattr(value=a, attr=_data) ['$124load_attr.3', 'a']
    $140call.5 = call $110load_global.0($124load_attr.3, i, func=$110load_global.0, args=[Var($124load_attr.3, unicode.py:295), Var(i, unicode.py:290)], kws=(), vararg=None, varkwarg=None, target=None) ['$110load_global.0', '$124load_attr.3', '$140call.5', 'i']
    $150return_value.6 = cast(value=$140call.5) ['$140call.5', '$150return_value.6']
    return $150return_value.6                ['$150return_value.6']
label 152:
    $154load_attr.1 = getattr(value=a, attr=_kind) ['$154load_attr.1', 'a']
    $164load_global.2 = global(PY_UNICODE_4BYTE_KIND: 4) ['$164load_global.2']
    $176compare_op.3 = $154load_attr.1 == $164load_global.2 ['$154load_attr.1', '$164load_global.2', '$176compare_op.3']
    bool182 = global(bool: <class 'bool'>)   ['bool182']
    $182pred = call bool182($176compare_op.3, func=bool182, args=(Var($176compare_op.3, unicode.py:296),), kws=(), vararg=None, varkwarg=None, target=None) ['$176compare_op.3', '$182pred', 'bool182']
    branch $182pred, 184, 226                ['$182pred']
label 184:
    $184load_global.0 = global(deref_uint32: <intrinsic deref_uint32>) ['$184load_global.0']
    $198load_attr.3 = getattr(value=a, attr=_data) ['$198load_attr.3', 'a']
    $214call.5 = call $184load_global.0($198load_attr.3, i, func=$184load_global.0, args=[Var($198load_attr.3, unicode.py:297), Var(i, unicode.py:290)], kws=(), vararg=None, varkwarg=None, target=None) ['$184load_global.0', '$198load_attr.3', '$214call.5', 'i']
    $224return_value.6 = cast(value=$214call.5) ['$214call.5', '$224return_value.6']
    return $224return_value.6                ['$224return_value.6']
label 226:
    $const226.0.1 = const(int, 0)            ['$const226.0.1']
    $228return_value.1 = cast(value=$const226.0.1) ['$228return_value.1', '$const226.0.1']
    return $228return_value.1                ['$228return_value.1']
label 0:
    a = arg(0, name=a)                       ['a']
    i = arg(1, name=i)                       ['i']
    ch = arg(2, name=ch)                     ['ch']
    $6load_attr.1 = getattr(value=a, attr=_kind) ['$6load_attr.1', 'a']
    $16load_global.2 = global(PY_UNICODE_1BYTE_KIND: 1) ['$16load_global.2']
    $28compare_op.3 = $6load_attr.1 == $16load_global.2 ['$16load_global.2', '$28compare_op.3', '$6load_attr.1']
    bool34 = global(bool: <class 'bool'>)    ['bool34']
    $34pred = call bool34($28compare_op.3, func=bool34, args=(Var($28compare_op.3, unicode.py:344),), kws=(), vararg=None, varkwarg=None, target=None) ['$28compare_op.3', '$34pred', 'bool34']
    branch $34pred, 36, 84                   ['$34pred']
label 36:
    $36load_global.0 = global(set_uint8: <intrinsic set_uint8>) ['$36load_global.0']
    $50load_attr.3 = getattr(value=a, attr=_data) ['$50load_attr.3', 'a']
    $68call.6 = call $36load_global.0($50load_attr.3, i, ch, func=$36load_global.0, args=[Var($50load_attr.3, unicode.py:345), Var(i, unicode.py:336), Var(ch, unicode.py:336)], kws=(), vararg=None, varkwarg=None, target=None) ['$36load_global.0', '$50load_attr.3', '$68call.6', 'ch', 'i']
    $const80.7.0 = const(NoneType, None)     ['$const80.7.0']
    $82return_value.8 = cast(value=$const80.7.0) ['$82return_value.8', '$const80.7.0']
    return $82return_value.8                 ['$82return_value.8']
label 84:
    $86load_attr.1 = getattr(value=a, attr=_kind) ['$86load_attr.1', 'a']
    $96load_global.2 = global(PY_UNICODE_2BYTE_KIND: 2) ['$96load_global.2']
    $108compare_op.3 = $86load_attr.1 == $96load_global.2 ['$108compare_op.3', '$86load_attr.1', '$96load_global.2']
    bool114 = global(bool: <class 'bool'>)   ['bool114']
    $114pred = call bool114($108compare_op.3, func=bool114, args=(Var($108compare_op.3, unicode.py:346),), kws=(), vararg=None, varkwarg=None, target=None) ['$108compare_op.3', '$114pred', 'bool114']
    branch $114pred, 116, 164                ['$114pred']
label 116:
    $116load_global.0 = global(set_uint16: <intrinsic set_uint16>) ['$116load_global.0']
    $130load_attr.3 = getattr(value=a, attr=_data) ['$130load_attr.3', 'a']
    $148call.6 = call $116load_global.0($130load_attr.3, i, ch, func=$116load_global.0, args=[Var($130load_attr.3, unicode.py:347), Var(i, unicode.py:336), Var(ch, unicode.py:336)], kws=(), vararg=None, varkwarg=None, target=None) ['$116load_global.0', '$130load_attr.3', '$148call.6', 'ch', 'i']
    $const160.7.0 = const(NoneType, None)    ['$const160.7.0']
    $162return_value.8 = cast(value=$const160.7.0) ['$162return_value.8', '$const160.7.0']
    return $162return_value.8                ['$162return_value.8']
label 164:
    $166load_attr.1 = getattr(value=a, attr=_kind) ['$166load_attr.1', 'a']
    $176load_global.2 = global(PY_UNICODE_4BYTE_KIND: 4) ['$176load_global.2']
    $188compare_op.3 = $166load_attr.1 == $176load_global.2 ['$166load_attr.1', '$176load_global.2', '$188compare_op.3']
    bool194 = global(bool: <class 'bool'>)   ['bool194']
    $194pred = call bool194($188compare_op.3, func=bool194, args=(Var($188compare_op.3, unicode.py:348),), kws=(), vararg=None, varkwarg=None, target=None) ['$188compare_op.3', '$194pred', 'bool194']
    branch $194pred, 196, 244                ['$194pred']
label 196:
    $196load_global.0 = global(set_uint32: <intrinsic set_uint32>) ['$196load_global.0']
    $210load_attr.3 = getattr(value=a, attr=_data) ['$210load_attr.3', 'a']
    $228call.6 = call $196load_global.0($210load_attr.3, i, ch, func=$196load_global.0, args=[Var($210load_attr.3, unicode.py:349), Var(i, unicode.py:336), Var(ch, unicode.py:336)], kws=(), vararg=None, varkwarg=None, target=None) ['$196load_global.0', '$210load_attr.3', '$228call.6', 'ch', 'i']
    $const240.7.0 = const(NoneType, None)    ['$const240.7.0']
    $242return_value.8 = cast(value=$const240.7.0) ['$242return_value.8', '$const240.7.0']
    return $242return_value.8                ['$242return_value.8']
label 244:
    $244load_global.0 = global(AssertionError: <class 'AssertionError'>) ['$244load_global.0']
    $const256.2.1 = const(str, Unexpected unicode representation in _set_code_point) ['$const256.2.1']
    $262call.3 = call $244load_global.0($const256.2.1, func=$244load_global.0, args=[Var($const256.2.1, unicode.py:352)], kws=(), vararg=None, varkwarg=None, target=None) ['$244load_global.0', '$262call.3', '$const256.2.1']
    raise $262call.3                         ['$262call.3']
label 0:
    obj = arg(0, name=obj)                   ['obj']
    attr = const(str, __repr__)              ['attr']
    $10load_global.1 = global(hasattr: <built-in function hasattr>) ['$10load_global.1']
    $30call.5 = call $10load_global.1(obj, attr, func=$10load_global.1, args=[Var(obj, old_builtins.py:1006), Var(attr, old_builtins.py:1007)], kws=(), vararg=None, varkwarg=None, target=None) ['$10load_global.1', '$30call.5', 'attr', 'obj']
    $const40.6.2 = const(bool, True)         ['$const40.6.2']
    $42compare_op.7 = $30call.5 == $const40.6.2 ['$30call.5', '$42compare_op.7', '$const40.6.2']
    bool48 = global(bool: <class 'bool'>)    ['bool48']
    $48pred = call bool48($42compare_op.7, func=bool48, args=(Var($42compare_op.7, old_builtins.py:1008),), kws=(), vararg=None, varkwarg=None, target=None) ['$42compare_op.7', '$48pred', 'bool48']
    branch $48pred, 50, 98                   ['$48pred']
label 50:
    $52load_global.1 = global(getattr: <built-in function getattr>) ['$52load_global.1']
    $72call.5 = call $52load_global.1(obj, attr, func=$52load_global.1, args=[Var(obj, old_builtins.py:1006), Var(attr, old_builtins.py:1007)], kws=(), vararg=None, varkwarg=None, target=None) ['$52load_global.1', '$72call.5', 'attr', 'obj']
    $86call.6 = call $72call.5(func=$72call.5, args=[], kws=(), vararg=None, varkwarg=None, target=None) ['$72call.5', '$86call.6']
    $96return_value.7 = cast(value=$86call.6) ['$86call.6', '$96return_value.7']
    return $96return_value.7                 ['$96return_value.7']
label 98:
    $98load_deref.0 = freevar(missing_repr_format: <object type:int64>) ['$98load_deref.0']
    $100return_value.1 = cast(value=$98load_deref.0) ['$100return_value.1', '$98load_deref.0']
    return $100return_value.1                ['$100return_value.1']
label 0:
    obj = arg(0, name=obj)                   ['obj']
    name = arg(1, name=name)                 ['name']
    $4load_global.0 = global(resolve_hasattr: <intrinsic resolve_hasattr>) ['$4load_global.0']
    $24call.4 = call $4load_global.0(obj, name, func=$4load_global.0, args=[Var(obj, old_builtins.py:998), Var(name, old_builtins.py:998)], kws=(), vararg=None, varkwarg=None, target=None) ['$24call.4', '$4load_global.0', 'name', 'obj']
    $34return_value.5 = cast(value=$24call.4) ['$24call.4', '$34return_value.5']
    return $34return_value.5                 ['$34return_value.5']
label 0:
    obj = arg(0, name=obj)                   ['obj']
    name = arg(1, name=name)                 ['name']
    $4load_global.0 = global(resolve_hasattr: <intrinsic resolve_hasattr>) ['$4load_global.0']
    $24call.4 = call $4load_global.0(obj, name, func=$4load_global.0, args=[Var(obj, old_builtins.py:998), Var(name, old_builtins.py:998)], kws=(), vararg=None, varkwarg=None, target=None) ['$24call.4', '$4load_global.0', 'name', 'obj']
    $34return_value.5 = cast(value=$24call.4) ['$24call.4', '$34return_value.5']
    return $34return_value.5                 ['$34return_value.5']
label 0:
    obj = arg(0, name=obj)                   ['obj']
    name = arg(1, name=name)                 ['name']
    $4load_global.0 = global(resolve_getattr: <intrinsic resolve_getattr>) ['$4load_global.0']
    $20load_global.4 = global(_getattr_default: _getattr_default_type()) ['$20load_global.4']
    $36call.5 = call $4load_global.0(obj, name, $20load_global.4, func=$4load_global.0, args=[Var(obj, old_builtins.py:954), Var(name, old_builtins.py:954), Var($20load_global.4, old_builtins.py:955)], kws=(), vararg=None, varkwarg=None, target=None) ['$20load_global.4', '$36call.5', '$4load_global.0', 'name', 'obj']
    $46return_value.6 = cast(value=$36call.5) ['$36call.5', '$46return_value.6']
    return $46return_value.6                 ['$46return_value.6']
label 0:
    n = arg(0, name=n)                       ['n']
    $6load_method.2 = getattr(value=n, attr=__str__) ['$6load_method.2', 'n']
    $32call.3 = call $6load_method.2(func=$6load_method.2, args=[], kws=(), vararg=None, varkwarg=None, target=None) ['$32call.3', '$6load_method.2']
    $42return_value.4 = cast(value=$32call.3) ['$32call.3', '$42return_value.4']
    return $42return_value.4                 ['$42return_value.4']
label 0:
    obj = arg(0, name=obj)                   ['obj']
    name = arg(1, name=name)                 ['name']
    $4load_global.0 = global(resolve_hasattr: <intrinsic resolve_hasattr>) ['$4load_global.0']
    $24call.4 = call $4load_global.0(obj, name, func=$4load_global.0, args=[Var(obj, old_builtins.py:998), Var(name, old_builtins.py:998)], kws=(), vararg=None, varkwarg=None, target=None) ['$24call.4', '$4load_global.0', 'name', 'obj']
    $34return_value.5 = cast(value=$24call.4) ['$24call.4', '$34return_value.5']
    return $34return_value.5                 ['$34return_value.5']
label 0:
    obj = arg(0, name=obj)                   ['obj']
    name = arg(1, name=name)                 ['name']
    $4load_global.0 = global(resolve_hasattr: <intrinsic resolve_hasattr>) ['$4load_global.0']
    $24call.4 = call $4load_global.0(obj, name, func=$4load_global.0, args=[Var(obj, old_builtins.py:998), Var(name, old_builtins.py:998)], kws=(), vararg=None, varkwarg=None, target=None) ['$24call.4', '$4load_global.0', 'name', 'obj']
    $34return_value.5 = cast(value=$24call.4) ['$24call.4', '$34return_value.5']
    return $34return_value.5                 ['$34return_value.5']
label 0:
    a = arg(0, name=a)                       ['a']
    b = arg(1, name=b)                       ['b']
    $6load_attr.1 = getattr(value=a, attr=_length) ['$6load_attr.1', 'a']
    $18load_attr.3 = getattr(value=b, attr=_length) ['$18load_attr.3', 'b']
    new_length = $6load_attr.1 + $18load_attr.3 ['$18load_attr.3', '$6load_attr.1', 'new_length']
    $34load_global.5 = global(_pick_kind: <function _pick_kind at 0x7fe1e54e59e0>) ['$34load_global.5']
    $48load_attr.8 = getattr(value=a, attr=_kind) ['$48load_attr.8', 'a']
    $60load_attr.10 = getattr(value=b, attr=_kind) ['$60load_attr.10', 'b']
    new_kind = call $34load_global.5($48load_attr.8, $60load_attr.10, func=$34load_global.5, args=[Var($48load_attr.8, unicode.py:1887), Var($60load_attr.10, unicode.py:1887)], kws=(), vararg=None, varkwarg=None, target=None) ['$34load_global.5', '$48load_attr.8', '$60load_attr.10', 'new_kind']
    $86load_global.12 = global(_pick_ascii: <function _pick_ascii at 0x7fe1e54e5b20>) ['$86load_global.12']
    $100load_attr.15 = getattr(value=a, attr=_is_ascii) ['$100load_attr.15', 'a']
    $112load_attr.17 = getattr(value=b, attr=_is_ascii) ['$112load_attr.17', 'b']
    new_ascii = call $86load_global.12($100load_attr.15, $112load_attr.17, func=$86load_global.12, args=[Var($100load_attr.15, unicode.py:1888), Var($112load_attr.17, unicode.py:1888)], kws=(), vararg=None, varkwarg=None, target=None) ['$100load_attr.15', '$112load_attr.17', '$86load_global.12', 'new_ascii']
    $138load_global.19 = global(_empty_string: <function _empty_string at 0x7fe1e54e54e0>) ['$138load_global.19']
    result = call $138load_global.19(new_kind, new_length, new_ascii, func=$138load_global.19, args=[Var(new_kind, unicode.py:1887), Var(new_length, unicode.py:1886), Var(new_ascii, unicode.py:1888)], kws=(), vararg=None, varkwarg=None, target=None) ['$138load_global.19', 'new_ascii', 'new_kind', 'new_length', 'result']
    $172load_global.25 = global(range: <class 'range'>) ['$172load_global.25']
    $184load_global.27 = global(len: <built-in function len>) ['$184load_global.27']
    $202call.30 = call $184load_global.27(a, func=$184load_global.27, args=[Var(a, unicode.py:1885)], kws=(), vararg=None, varkwarg=None, target=None) ['$184load_global.27', '$202call.30', 'a']
    $216call.31 = call $172load_global.25($202call.30, func=$172load_global.25, args=[Var($202call.30, unicode.py:1890)], kws=(), vararg=None, varkwarg=None, target=None) ['$172load_global.25', '$202call.30', '$216call.31']
    $226get_iter.32 = getiter(value=$216call.31) ['$216call.31', '$226get_iter.32']
    $phi228.0 = $226get_iter.32              ['$226get_iter.32', '$phi228.0']
    jump 228                                 []
label 228:
    $228for_iter.1 = iternext(value=$phi228.0) ['$228for_iter.1', '$phi228.0']
    $228for_iter.2 = pair_first(value=$228for_iter.1) ['$228for_iter.1', '$228for_iter.2']
    $228for_iter.3 = pair_second(value=$228for_iter.1) ['$228for_iter.1', '$228for_iter.3']
    $phi230.1 = $228for_iter.2               ['$228for_iter.2', '$phi230.1']
    branch $228for_iter.3, 230, 296          ['$228for_iter.3']
label 230:
    i = $phi230.1                            ['$phi230.1', 'i']
    $232load_global.2 = global(_set_code_point: <function _set_code_point at 0x7fe1e54e5a80>) ['$232load_global.2']
    $248load_global.6 = global(_get_code_point: <function _get_code_point at 0x7fe1e54e5620>) ['$248load_global.6']
    $268call.10 = call $248load_global.6(a, i, func=$248load_global.6, args=[Var(a, unicode.py:1885), Var(i, unicode.py:1890)], kws=(), vararg=None, varkwarg=None, target=None) ['$248load_global.6', '$268call.10', 'a', 'i']
    $282call.11 = call $232load_global.2(result, i, $268call.10, func=$232load_global.2, args=[Var(result, unicode.py:1889), Var(i, unicode.py:1890), Var($268call.10, unicode.py:1891)], kws=(), vararg=None, varkwarg=None, target=None) ['$232load_global.2', '$268call.10', '$282call.11', 'i', 'result']
    jump 228                                 []
label 296:
    $296load_global.0 = global(range: <class 'range'>) ['$296load_global.0']
    $308load_global.2 = global(len: <built-in function len>) ['$308load_global.2']
    $326call.5 = call $308load_global.2(b, func=$308load_global.2, args=[Var(b, unicode.py:1885)], kws=(), vararg=None, varkwarg=None, target=None) ['$308load_global.2', '$326call.5', 'b']
    $340call.6 = call $296load_global.0($326call.5, func=$296load_global.0, args=[Var($326call.5, unicode.py:1892)], kws=(), vararg=None, varkwarg=None, target=None) ['$296load_global.0', '$326call.5', '$340call.6']
    $350get_iter.7 = getiter(value=$340call.6) ['$340call.6', '$350get_iter.7']
    $phi352.0 = $350get_iter.7               ['$350get_iter.7', '$phi352.0']
    jump 352                                 []
label 352:
    $352for_iter.1 = iternext(value=$phi352.0) ['$352for_iter.1', '$phi352.0']
    $352for_iter.2 = pair_first(value=$352for_iter.1) ['$352for_iter.1', '$352for_iter.2']
    $352for_iter.3 = pair_second(value=$352for_iter.1) ['$352for_iter.1', '$352for_iter.3']
    $phi354.1 = $352for_iter.2               ['$352for_iter.2', '$phi354.1']
    branch $352for_iter.3, 354, 452          ['$352for_iter.3']
label 354:
    j = $phi354.1                            ['$phi354.1', 'j']
    $356load_global.2 = global(_set_code_point: <function _set_code_point at 0x7fe1e54e5a80>) ['$356load_global.2']
    $370load_global.5 = global(len: <built-in function len>) ['$370load_global.5']
    $388call.8 = call $370load_global.5(a, func=$370load_global.5, args=[Var(a, unicode.py:1885)], kws=(), vararg=None, varkwarg=None, target=None) ['$370load_global.5', '$388call.8', 'a']
    $binop_add400.10 = $388call.8 + j        ['$388call.8', '$binop_add400.10', 'j']
    $404load_global.11 = global(_get_code_point: <function _get_code_point at 0x7fe1e54e5620>) ['$404load_global.11']
    $424call.15 = call $404load_global.11(b, j, func=$404load_global.11, args=[Var(b, unicode.py:1885), Var(j, unicode.py:1892)], kws=(), vararg=None, varkwarg=None, target=None) ['$404load_global.11', '$424call.15', 'b', 'j']
    $438call.16 = call $356load_global.2(result, $binop_add400.10, $424call.15, func=$356load_global.2, args=[Var(result, unicode.py:1889), Var($binop_add400.10, unicode.py:1893), Var($424call.15, unicode.py:1893)], kws=(), vararg=None, varkwarg=None, target=None) ['$356load_global.2', '$424call.15', '$438call.16', '$binop_add400.10', 'result']
    jump 352                                 []
label 452:
    $454return_value.1 = cast(value=result)  ['$454return_value.1', 'result']
    return $454return_value.1                ['$454return_value.1']
label 0:
    kind1 = arg(0, name=kind1)               ['kind1']
    kind2 = arg(1, name=kind2)               ['kind2']
    $6load_global.1 = global(PY_UNICODE_WCHAR_KIND: 0) ['$6load_global.1']
    $18compare_op.2 = kind1 == $6load_global.1 ['$18compare_op.2', '$6load_global.1', 'kind1']
    bool24 = global(bool: <class 'bool'>)    ['bool24']
    $24pred = call bool24($18compare_op.2, func=bool24, args=(Var($18compare_op.2, unicode.py:373),), kws=(), vararg=None, varkwarg=None, target=None) ['$18compare_op.2', '$24pred', 'bool24']
    branch $24pred, 48, 26                   ['$24pred']
label 26:
    $28load_global.1 = global(PY_UNICODE_WCHAR_KIND: 0) ['$28load_global.1']
    $40compare_op.2 = kind2 == $28load_global.1 ['$28load_global.1', '$40compare_op.2', 'kind2']
    bool46 = global(bool: <class 'bool'>)    ['bool46']
    $46pred = call bool46($40compare_op.2, func=bool46, args=(Var($40compare_op.2, unicode.py:373),), kws=(), vararg=None, varkwarg=None, target=None) ['$40compare_op.2', '$46pred', 'bool46']
    branch $46pred, 48, 78                   ['$46pred']
label 48:
    $48load_global.0 = global(AssertionError: <class 'AssertionError'>) ['$48load_global.0']
    $const60.2.1 = const(str, PY_UNICODE_WCHAR_KIND unsupported) ['$const60.2.1']
    $66call.3 = call $48load_global.0($const60.2.1, func=$48load_global.0, args=[Var($const60.2.1, unicode.py:374)], kws=(), vararg=None, varkwarg=None, target=None) ['$48load_global.0', '$66call.3', '$const60.2.1']
    raise $66call.3                          ['$66call.3']
label 78:
    $80load_global.1 = global(PY_UNICODE_1BYTE_KIND: 1) ['$80load_global.1']
    $92compare_op.2 = kind1 == $80load_global.1 ['$80load_global.1', '$92compare_op.2', 'kind1']
    bool98 = global(bool: <class 'bool'>)    ['bool98']
    $98pred = call bool98($92compare_op.2, func=bool98, args=(Var($92compare_op.2, unicode.py:376),), kws=(), vararg=None, varkwarg=None, target=None) ['$92compare_op.2', '$98pred', 'bool98']
    branch $98pred, 100, 104                 ['$98pred']
label 100:
    $102return_value.1 = cast(value=kind2)   ['$102return_value.1', 'kind2']
    return $102return_value.1                ['$102return_value.1']
label 104:
    $106load_global.1 = global(PY_UNICODE_2BYTE_KIND: 2) ['$106load_global.1']
    $118compare_op.2 = kind1 == $106load_global.1 ['$106load_global.1', '$118compare_op.2', 'kind1']
    bool124 = global(bool: <class 'bool'>)   ['bool124']
    $124pred = call bool124($118compare_op.2, func=bool124, args=(Var($118compare_op.2, unicode.py:378),), kws=(), vararg=None, varkwarg=None, target=None) ['$118compare_op.2', '$124pred', 'bool124']
    branch $124pred, 126, 156                ['$124pred']
label 126:
    $128load_global.1 = global(PY_UNICODE_4BYTE_KIND: 4) ['$128load_global.1']
    $140compare_op.2 = kind2 == $128load_global.1 ['$128load_global.1', '$140compare_op.2', 'kind2']
    bool146 = global(bool: <class 'bool'>)   ['bool146']
    $146pred = call bool146($140compare_op.2, func=bool146, args=(Var($140compare_op.2, unicode.py:379),), kws=(), vararg=None, varkwarg=None, target=None) ['$140compare_op.2', '$146pred', 'bool146']
    branch $146pred, 148, 152                ['$146pred']
label 148:
    $150return_value.1 = cast(value=kind2)   ['$150return_value.1', 'kind2']
    return $150return_value.1                ['$150return_value.1']
label 152:
    $154return_value.1 = cast(value=kind1)   ['$154return_value.1', 'kind1']
    return $154return_value.1                ['$154return_value.1']
label 156:
    $158load_global.1 = global(PY_UNICODE_4BYTE_KIND: 4) ['$158load_global.1']
    $170compare_op.2 = kind1 == $158load_global.1 ['$158load_global.1', '$170compare_op.2', 'kind1']
    bool176 = global(bool: <class 'bool'>)   ['bool176']
    $176pred = call bool176($170compare_op.2, func=bool176, args=(Var($170compare_op.2, unicode.py:383),), kws=(), vararg=None, varkwarg=None, target=None) ['$170compare_op.2', '$176pred', 'bool176']
    branch $176pred, 178, 182                ['$176pred']
label 178:
    $180return_value.1 = cast(value=kind1)   ['$180return_value.1', 'kind1']
    return $180return_value.1                ['$180return_value.1']
label 182:
    $182load_global.0 = global(AssertionError: <class 'AssertionError'>) ['$182load_global.0']
    $const194.2.2 = const(str, Unexpected unicode representation in _pick_kind) ['$const194.2.2']
    $200call.3 = call $182load_global.0($const194.2.2, func=$182load_global.0, args=[Var($const194.2.2, unicode.py:387)], kws=(), vararg=None, varkwarg=None, target=None) ['$182load_global.0', '$200call.3', '$const194.2.2']
    raise $200call.3                         ['$200call.3']
label 0:
    is_ascii1 = arg(0, name=is_ascii1)       ['is_ascii1']
    is_ascii2 = arg(1, name=is_ascii2)       ['is_ascii2']
    $const6.1.1 = const(int, 1)              ['$const6.1.1']
    $8compare_op.2 = is_ascii1 == $const6.1.1 ['$8compare_op.2', '$const6.1.1', 'is_ascii1']
    bool14 = global(bool: <class 'bool'>)    ['bool14']
    $14pred = call bool14($8compare_op.2, func=bool14, args=(Var($8compare_op.2, unicode.py:394),), kws=(), vararg=None, varkwarg=None, target=None) ['$14pred', '$8compare_op.2', 'bool14']
    branch $14pred, 16, 68                   ['$14pred']
label 16:
    $const18.1.1 = const(int, 1)             ['$const18.1.1']
    $20compare_op.2 = is_ascii2 == $const18.1.1 ['$20compare_op.2', '$const18.1.1', 'is_ascii2']
    bool26 = global(bool: <class 'bool'>)    ['bool26']
    $26pred = call bool26($20compare_op.2, func=bool26, args=(Var($20compare_op.2, unicode.py:394),), kws=(), vararg=None, varkwarg=None, target=None) ['$20compare_op.2', '$26pred', 'bool26']
    branch $26pred, 28, 68                   ['$26pred']
label 28:
    $28load_global.0 = global(types: <module 'numba.core.types' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/core/types/__init__.py'>) ['$28load_global.0']
    $40load_attr.2 = getattr(value=$28load_global.0, attr=uint32) ['$28load_global.0', '$40load_attr.2']
    $const50.3.1 = const(int, 1)             ['$const50.3.1']
    $56call.4 = call $40load_attr.2($const50.3.1, func=$40load_attr.2, args=[Var($const50.3.1, unicode.py:395)], kws=(), vararg=None, varkwarg=None, target=None) ['$40load_attr.2', '$56call.4', '$const50.3.1']
    $66return_value.5 = cast(value=$56call.4) ['$56call.4', '$66return_value.5']
    return $66return_value.5                 ['$66return_value.5']
label 68:
    $68load_global.0 = global(types: <module 'numba.core.types' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/core/types/__init__.py'>) ['$68load_global.0']
    $80load_attr.2 = getattr(value=$68load_global.0, attr=uint32) ['$68load_global.0', '$80load_attr.2']
    $const90.3.2 = const(int, 0)             ['$const90.3.2']
    $96call.4 = call $80load_attr.2($const90.3.2, func=$80load_attr.2, args=[Var($const90.3.2, unicode.py:396)], kws=(), vararg=None, varkwarg=None, target=None) ['$80load_attr.2', '$96call.4', '$const90.3.2']
    $106return_value.5 = cast(value=$96call.4) ['$106return_value.5', '$96call.4']
    return $106return_value.5                ['$106return_value.5']
label 0:
    kind = arg(0, name=kind)                 ['kind']
    length = arg(1, name=length)             ['length']
    is_ascii = arg(2, name=is_ascii)         ['is_ascii']
    $4load_global.0 = global(_kind_to_byte_width: <function _kind_to_byte_width at 0x7fe1e54e5c60>) ['$4load_global.0']
    char_width = call $4load_global.0(kind, func=$4load_global.0, args=[Var(kind, unicode.py:281)], kws=(), vararg=None, varkwarg=None, target=None) ['$4load_global.0', 'char_width', 'kind']
    $34load_global.4 = global(_malloc_string: <intrinsic _malloc_string>) ['$34load_global.4']
    s = call $34load_global.4(kind, char_width, length, is_ascii, func=$34load_global.4, args=[Var(kind, unicode.py:281), Var(char_width, unicode.py:283), Var(length, unicode.py:281), Var(is_ascii, unicode.py:281)], kws=(), vararg=None, varkwarg=None, target=None) ['$34load_global.4', 'char_width', 'is_ascii', 'kind', 'length', 's']
    $70load_global.11 = global(_set_code_point: <function _set_code_point at 0x7fe1e54e5a80>) ['$70load_global.11']
    $86load_global.15 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$86load_global.15']
    $98load_attr.17 = getattr(value=$86load_global.15, attr=uint32) ['$86load_global.15', '$98load_attr.17']
    $const108.18.1 = const(int, 0)           ['$const108.18.1']
    $114call.19 = call $98load_attr.17($const108.18.1, func=$98load_attr.17, args=[Var($const108.18.1, unicode.py:285)], kws=(), vararg=None, varkwarg=None, target=None) ['$114call.19', '$98load_attr.17', '$const108.18.1']
    $128call.20 = call $70load_global.11(s, length, $114call.19, func=$70load_global.11, args=[Var(s, unicode.py:284), Var(length, unicode.py:281), Var($114call.19, unicode.py:285)], kws=(), vararg=None, varkwarg=None, target=None) ['$114call.19', '$128call.20', '$70load_global.11', 'length', 's']
    $142return_value.22 = cast(value=s)      ['$142return_value.22', 's']
    return $142return_value.22               ['$142return_value.22']
label 0:
    kind = arg(0, name=kind)                 ['kind']
    $6load_global.1 = global(PY_UNICODE_1BYTE_KIND: 1) ['$6load_global.1']
    $18compare_op.2 = kind == $6load_global.1 ['$18compare_op.2', '$6load_global.1', 'kind']
    bool24 = global(bool: <class 'bool'>)    ['bool24']
    $24pred = call bool24($18compare_op.2, func=bool24, args=(Var($18compare_op.2, unicode.py:413),), kws=(), vararg=None, varkwarg=None, target=None) ['$18compare_op.2', '$24pred', 'bool24']
    branch $24pred, 26, 30                   ['$24pred']
label 26:
    $const26.0.1 = const(int, 1)             ['$const26.0.1']
    $28return_value.1 = cast(value=$const26.0.1) ['$28return_value.1', '$const26.0.1']
    return $28return_value.1                 ['$28return_value.1']
label 30:
    $32load_global.1 = global(PY_UNICODE_2BYTE_KIND: 2) ['$32load_global.1']
    $44compare_op.2 = kind == $32load_global.1 ['$32load_global.1', '$44compare_op.2', 'kind']
    bool50 = global(bool: <class 'bool'>)    ['bool50']
    $50pred = call bool50($44compare_op.2, func=bool50, args=(Var($44compare_op.2, unicode.py:415),), kws=(), vararg=None, varkwarg=None, target=None) ['$44compare_op.2', '$50pred', 'bool50']
    branch $50pred, 52, 56                   ['$50pred']
label 52:
    $const52.0.2 = const(int, 2)             ['$const52.0.2']
    $54return_value.1 = cast(value=$const52.0.2) ['$54return_value.1', '$const52.0.2']
    return $54return_value.1                 ['$54return_value.1']
label 56:
    $58load_global.1 = global(PY_UNICODE_4BYTE_KIND: 4) ['$58load_global.1']
    $70compare_op.2 = kind == $58load_global.1 ['$58load_global.1', '$70compare_op.2', 'kind']
    bool76 = global(bool: <class 'bool'>)    ['bool76']
    $76pred = call bool76($70compare_op.2, func=bool76, args=(Var($70compare_op.2, unicode.py:417),), kws=(), vararg=None, varkwarg=None, target=None) ['$70compare_op.2', '$76pred', 'bool76']
    branch $76pred, 78, 82                   ['$76pred']
label 78:
    $const78.0.3 = const(int, 4)             ['$const78.0.3']
    $80return_value.1 = cast(value=$const78.0.3) ['$80return_value.1', '$const78.0.3']
    return $80return_value.1                 ['$80return_value.1']
label 82:
    $84load_global.1 = global(PY_UNICODE_WCHAR_KIND: 0) ['$84load_global.1']
    $96compare_op.2 = kind == $84load_global.1 ['$84load_global.1', '$96compare_op.2', 'kind']
    bool102 = global(bool: <class 'bool'>)   ['bool102']
    $102pred = call bool102($96compare_op.2, func=bool102, args=(Var($96compare_op.2, unicode.py:419),), kws=(), vararg=None, varkwarg=None, target=None) ['$102pred', '$96compare_op.2', 'bool102']
    branch $102pred, 104, 134                ['$102pred']
label 104:
    $104load_global.0 = global(AssertionError: <class 'AssertionError'>) ['$104load_global.0']
    $const116.2.4 = const(str, PY_UNICODE_WCHAR_KIND unsupported) ['$const116.2.4']
    $122call.3 = call $104load_global.0($const116.2.4, func=$104load_global.0, args=[Var($const116.2.4, unicode.py:420)], kws=(), vararg=None, varkwarg=None, target=None) ['$104load_global.0', '$122call.3', '$const116.2.4']
    raise $122call.3                         ['$122call.3']
label 134:
    $134load_global.0 = global(AssertionError: <class 'AssertionError'>) ['$134load_global.0']
    $const146.2.5 = const(str, Unexpected unicode encoding encountered) ['$const146.2.5']
    $152call.3 = call $134load_global.0($const146.2.5, func=$134load_global.0, args=[Var($const146.2.5, unicode.py:422)], kws=(), vararg=None, varkwarg=None, target=None) ['$134load_global.0', '$152call.3', '$const146.2.5']
    raise $152call.3                         ['$152call.3']
label 0:
    a = arg(0, name=a)                       ['a']
    i = arg(1, name=i)                       ['i']
    ch = arg(2, name=ch)                     ['ch']
    $6load_attr.1 = getattr(value=a, attr=_kind) ['$6load_attr.1', 'a']
    $16load_global.2 = global(PY_UNICODE_1BYTE_KIND: 1) ['$16load_global.2']
    $28compare_op.3 = $6load_attr.1 == $16load_global.2 ['$16load_global.2', '$28compare_op.3', '$6load_attr.1']
    bool34 = global(bool: <class 'bool'>)    ['bool34']
    $34pred = call bool34($28compare_op.3, func=bool34, args=(Var($28compare_op.3, unicode.py:344),), kws=(), vararg=None, varkwarg=None, target=None) ['$28compare_op.3', '$34pred', 'bool34']
    branch $34pred, 36, 84                   ['$34pred']
label 36:
    $36load_global.0 = global(set_uint8: <intrinsic set_uint8>) ['$36load_global.0']
    $50load_attr.3 = getattr(value=a, attr=_data) ['$50load_attr.3', 'a']
    $68call.6 = call $36load_global.0($50load_attr.3, i, ch, func=$36load_global.0, args=[Var($50load_attr.3, unicode.py:345), Var(i, unicode.py:336), Var(ch, unicode.py:336)], kws=(), vararg=None, varkwarg=None, target=None) ['$36load_global.0', '$50load_attr.3', '$68call.6', 'ch', 'i']
    $const80.7.0 = const(NoneType, None)     ['$const80.7.0']
    $82return_value.8 = cast(value=$const80.7.0) ['$82return_value.8', '$const80.7.0']
    return $82return_value.8                 ['$82return_value.8']
label 84:
    $86load_attr.1 = getattr(value=a, attr=_kind) ['$86load_attr.1', 'a']
    $96load_global.2 = global(PY_UNICODE_2BYTE_KIND: 2) ['$96load_global.2']
    $108compare_op.3 = $86load_attr.1 == $96load_global.2 ['$108compare_op.3', '$86load_attr.1', '$96load_global.2']
    bool114 = global(bool: <class 'bool'>)   ['bool114']
    $114pred = call bool114($108compare_op.3, func=bool114, args=(Var($108compare_op.3, unicode.py:346),), kws=(), vararg=None, varkwarg=None, target=None) ['$108compare_op.3', '$114pred', 'bool114']
    branch $114pred, 116, 164                ['$114pred']
label 116:
    $116load_global.0 = global(set_uint16: <intrinsic set_uint16>) ['$116load_global.0']
    $130load_attr.3 = getattr(value=a, attr=_data) ['$130load_attr.3', 'a']
    $148call.6 = call $116load_global.0($130load_attr.3, i, ch, func=$116load_global.0, args=[Var($130load_attr.3, unicode.py:347), Var(i, unicode.py:336), Var(ch, unicode.py:336)], kws=(), vararg=None, varkwarg=None, target=None) ['$116load_global.0', '$130load_attr.3', '$148call.6', 'ch', 'i']
    $const160.7.0 = const(NoneType, None)    ['$const160.7.0']
    $162return_value.8 = cast(value=$const160.7.0) ['$162return_value.8', '$const160.7.0']
    return $162return_value.8                ['$162return_value.8']
label 164:
    $166load_attr.1 = getattr(value=a, attr=_kind) ['$166load_attr.1', 'a']
    $176load_global.2 = global(PY_UNICODE_4BYTE_KIND: 4) ['$176load_global.2']
    $188compare_op.3 = $166load_attr.1 == $176load_global.2 ['$166load_attr.1', '$176load_global.2', '$188compare_op.3']
    bool194 = global(bool: <class 'bool'>)   ['bool194']
    $194pred = call bool194($188compare_op.3, func=bool194, args=(Var($188compare_op.3, unicode.py:348),), kws=(), vararg=None, varkwarg=None, target=None) ['$188compare_op.3', '$194pred', 'bool194']
    branch $194pred, 196, 244                ['$194pred']
label 196:
    $196load_global.0 = global(set_uint32: <intrinsic set_uint32>) ['$196load_global.0']
    $210load_attr.3 = getattr(value=a, attr=_data) ['$210load_attr.3', 'a']
    $228call.6 = call $196load_global.0($210load_attr.3, i, ch, func=$196load_global.0, args=[Var($210load_attr.3, unicode.py:349), Var(i, unicode.py:336), Var(ch, unicode.py:336)], kws=(), vararg=None, varkwarg=None, target=None) ['$196load_global.0', '$210load_attr.3', '$228call.6', 'ch', 'i']
    $const240.7.0 = const(NoneType, None)    ['$const240.7.0']
    $242return_value.8 = cast(value=$const240.7.0) ['$242return_value.8', '$const240.7.0']
    return $242return_value.8                ['$242return_value.8']
label 244:
    $244load_global.0 = global(AssertionError: <class 'AssertionError'>) ['$244load_global.0']
    $const256.2.1 = const(str, Unexpected unicode representation in _set_code_point) ['$const256.2.1']
    $262call.3 = call $244load_global.0($const256.2.1, func=$244load_global.0, args=[Var($const256.2.1, unicode.py:352)], kws=(), vararg=None, varkwarg=None, target=None) ['$244load_global.0', '$262call.3', '$const256.2.1']
    raise $262call.3                         ['$262call.3']
label 0:
    object = arg(0, name=object)             ['object']
    attr = const(str, __str__)               ['attr']
    $8load_global.1 = global(hasattr: <built-in function hasattr>) ['$8load_global.1']
    $28call.5 = call $8load_global.1(object, attr, func=$8load_global.1, args=[Var(object, old_builtins.py:1019), Var(attr, old_builtins.py:1020)], kws=(), vararg=None, varkwarg=None, target=None) ['$28call.5', '$8load_global.1', 'attr', 'object']
    $const38.6.2 = const(bool, True)         ['$const38.6.2']
    $40compare_op.7 = $28call.5 == $const38.6.2 ['$28call.5', '$40compare_op.7', '$const38.6.2']
    bool46 = global(bool: <class 'bool'>)    ['bool46']
    $46pred = call bool46($40compare_op.7, func=bool46, args=(Var($40compare_op.7, old_builtins.py:1021),), kws=(), vararg=None, varkwarg=None, target=None) ['$40compare_op.7', '$46pred', 'bool46']
    branch $46pred, 48, 96                   ['$46pred']
label 48:
    $50load_global.1 = global(getattr: <built-in function getattr>) ['$50load_global.1']
    $70call.5 = call $50load_global.1(object, attr, func=$50load_global.1, args=[Var(object, old_builtins.py:1019), Var(attr, old_builtins.py:1020)], kws=(), vararg=None, varkwarg=None, target=None) ['$50load_global.1', '$70call.5', 'attr', 'object']
    $84call.6 = call $70call.5(func=$70call.5, args=[], kws=(), vararg=None, varkwarg=None, target=None) ['$70call.5', '$84call.6']
    $94return_value.7 = cast(value=$84call.6) ['$84call.6', '$94return_value.7']
    return $94return_value.7                 ['$94return_value.7']
label 96:
    $96load_global.0 = global(repr: <built-in function repr>) ['$96load_global.0']
    $114call.3 = call $96load_global.0(object, func=$96load_global.0, args=[Var(object, old_builtins.py:1019)], kws=(), vararg=None, varkwarg=None, target=None) ['$114call.3', '$96load_global.0', 'object']
    $124return_value.4 = cast(value=$114call.3) ['$114call.3', '$124return_value.4']
    return $124return_value.4                ['$124return_value.4']
label 0:
    obj = arg(0, name=obj)                   ['obj']
    name = arg(1, name=name)                 ['name']
    $4load_global.0 = global(resolve_hasattr: <intrinsic resolve_hasattr>) ['$4load_global.0']
    $24call.4 = call $4load_global.0(obj, name, func=$4load_global.0, args=[Var(obj, old_builtins.py:998), Var(name, old_builtins.py:998)], kws=(), vararg=None, varkwarg=None, target=None) ['$24call.4', '$4load_global.0', 'name', 'obj']
    $34return_value.5 = cast(value=$24call.4) ['$24call.4', '$34return_value.5']
    return $34return_value.5                 ['$34return_value.5']
label 0:
    obj = arg(0, name=obj)                   ['obj']
    name = arg(1, name=name)                 ['name']
    $4load_global.0 = global(resolve_hasattr: <intrinsic resolve_hasattr>) ['$4load_global.0']
    $24call.4 = call $4load_global.0(obj, name, func=$4load_global.0, args=[Var(obj, old_builtins.py:998), Var(name, old_builtins.py:998)], kws=(), vararg=None, varkwarg=None, target=None) ['$24call.4', '$4load_global.0', 'name', 'obj']
    $34return_value.5 = cast(value=$24call.4) ['$24call.4', '$34return_value.5']
    return $34return_value.5                 ['$34return_value.5']
label 0:
    obj = arg(0, name=obj)                   ['obj']
    name = arg(1, name=name)                 ['name']
    $4load_global.0 = global(resolve_getattr: <intrinsic resolve_getattr>) ['$4load_global.0']
    $20load_global.4 = global(_getattr_default: _getattr_default_type()) ['$20load_global.4']
    $36call.5 = call $4load_global.0(obj, name, $20load_global.4, func=$4load_global.0, args=[Var(obj, old_builtins.py:954), Var(name, old_builtins.py:954), Var($20load_global.4, old_builtins.py:955)], kws=(), vararg=None, varkwarg=None, target=None) ['$20load_global.4', '$36call.5', '$4load_global.0', 'name', 'obj']
    $46return_value.6 = cast(value=$36call.5) ['$36call.5', '$46return_value.6']
    return $46return_value.6                 ['$46return_value.6']
label 0:
    s = arg(0, name=s)                       ['s']
    $6return_value.1 = cast(value=s)         ['$6return_value.1', 's']
    return $6return_value.1                  ['$6return_value.1']
label 0:
    obj = arg(0, name=obj)                   ['obj']
    attr = const(str, __repr__)              ['attr']
    $10load_global.1 = global(hasattr: <built-in function hasattr>) ['$10load_global.1']
    $30call.5 = call $10load_global.1(obj, attr, func=$10load_global.1, args=[Var(obj, old_builtins.py:1006), Var(attr, old_builtins.py:1007)], kws=(), vararg=None, varkwarg=None, target=None) ['$10load_global.1', '$30call.5', 'attr', 'obj']
    $const40.6.2 = const(bool, True)         ['$const40.6.2']
    $42compare_op.7 = $30call.5 == $const40.6.2 ['$30call.5', '$42compare_op.7', '$const40.6.2']
    bool48 = global(bool: <class 'bool'>)    ['bool48']
    $48pred = call bool48($42compare_op.7, func=bool48, args=(Var($42compare_op.7, old_builtins.py:1008),), kws=(), vararg=None, varkwarg=None, target=None) ['$42compare_op.7', '$48pred', 'bool48']
    branch $48pred, 50, 98                   ['$48pred']
label 50:
    $52load_global.1 = global(getattr: <built-in function getattr>) ['$52load_global.1']
    $72call.5 = call $52load_global.1(obj, attr, func=$52load_global.1, args=[Var(obj, old_builtins.py:1006), Var(attr, old_builtins.py:1007)], kws=(), vararg=None, varkwarg=None, target=None) ['$52load_global.1', '$72call.5', 'attr', 'obj']
    $86call.6 = call $72call.5(func=$72call.5, args=[], kws=(), vararg=None, varkwarg=None, target=None) ['$72call.5', '$86call.6']
    $96return_value.7 = cast(value=$86call.6) ['$86call.6', '$96return_value.7']
    return $96return_value.7                 ['$96return_value.7']
label 98:
    $98load_deref.0 = freevar(missing_repr_format: <object type:unicode_type>) ['$98load_deref.0']
    $100return_value.1 = cast(value=$98load_deref.0) ['$100return_value.1', '$98load_deref.0']
    return $100return_value.1                ['$100return_value.1']
label 0:
    obj = arg(0, name=obj)                   ['obj']
    name = arg(1, name=name)                 ['name']
    $4load_global.0 = global(resolve_hasattr: <intrinsic resolve_hasattr>) ['$4load_global.0']
    $24call.4 = call $4load_global.0(obj, name, func=$4load_global.0, args=[Var(obj, old_builtins.py:998), Var(name, old_builtins.py:998)], kws=(), vararg=None, varkwarg=None, target=None) ['$24call.4', '$4load_global.0', 'name', 'obj']
    $34return_value.5 = cast(value=$24call.4) ['$24call.4', '$34return_value.5']
    return $34return_value.5                 ['$34return_value.5']
label 0:
    obj = arg(0, name=obj)                   ['obj']
    name = arg(1, name=name)                 ['name']
    $4load_global.0 = global(resolve_hasattr: <intrinsic resolve_hasattr>) ['$4load_global.0']
    $24call.4 = call $4load_global.0(obj, name, func=$4load_global.0, args=[Var(obj, old_builtins.py:998), Var(name, old_builtins.py:998)], kws=(), vararg=None, varkwarg=None, target=None) ['$24call.4', '$4load_global.0', 'name', 'obj']
    $34return_value.5 = cast(value=$24call.4) ['$24call.4', '$34return_value.5']
    return $34return_value.5                 ['$34return_value.5']
label 0:
    obj = arg(0, name=obj)                   ['obj']
    name = arg(1, name=name)                 ['name']
    $4load_global.0 = global(resolve_getattr: <intrinsic resolve_getattr>) ['$4load_global.0']
    $20load_global.4 = global(_getattr_default: _getattr_default_type()) ['$20load_global.4']
    $36call.5 = call $4load_global.0(obj, name, $20load_global.4, func=$4load_global.0, args=[Var(obj, old_builtins.py:954), Var(name, old_builtins.py:954), Var($20load_global.4, old_builtins.py:955)], kws=(), vararg=None, varkwarg=None, target=None) ['$20load_global.4', '$36call.5', '$4load_global.0', 'name', 'obj']
    $46return_value.6 = cast(value=$36call.5) ['$36call.5', '$46return_value.6']
    return $46return_value.6                 ['$46return_value.6']
label 0:
    s = arg(0, name=s)                       ['s']
    $const4.0.1 = const(str, ')              ['$const4.0.1']
    $binop_add8.2 = $const4.0.1 + s          ['$binop_add8.2', '$const4.0.1', 's']
    $const12.3.1 = const(str, ')             ['$const12.3.1']
    $binop_add14.4 = $binop_add8.2 + $const12.3.1 ['$binop_add14.4', '$binop_add8.2', '$const12.3.1']
    $18return_value.5 = cast(value=$binop_add14.4) ['$18return_value.5', '$binop_add14.4']
    return $18return_value.5                 ['$18return_value.5']
label 0:
    obj = arg(0, name=obj)                   ['obj']
    name = arg(1, name=name)                 ['name']
    $4load_global.0 = global(resolve_hasattr: <intrinsic resolve_hasattr>) ['$4load_global.0']
    $24call.4 = call $4load_global.0(obj, name, func=$4load_global.0, args=[Var(obj, old_builtins.py:998), Var(name, old_builtins.py:998)], kws=(), vararg=None, varkwarg=None, target=None) ['$24call.4', '$4load_global.0', 'name', 'obj']
    $34return_value.5 = cast(value=$24call.4) ['$24call.4', '$34return_value.5']
    return $34return_value.5                 ['$34return_value.5']
label 0:
    obj = arg(0, name=obj)                   ['obj']
    name = arg(1, name=name)                 ['name']
    $4load_global.0 = global(resolve_hasattr: <intrinsic resolve_hasattr>) ['$4load_global.0']
    $24call.4 = call $4load_global.0(obj, name, func=$4load_global.0, args=[Var(obj, old_builtins.py:998), Var(name, old_builtins.py:998)], kws=(), vararg=None, varkwarg=None, target=None) ['$24call.4', '$4load_global.0', 'name', 'obj']
    $34return_value.5 = cast(value=$24call.4) ['$24call.4', '$34return_value.5']
    return $34return_value.5                 ['$34return_value.5']
label 0:
    data = arg(0, name=data)                 ['data']
    n_neighbors = arg(1, name=n_neighbors)   ['n_neighbors']
    rng_state = arg(2, name=rng_state)       ['rng_state']
    max_candidates = arg(3, name=max_candidates) ['max_candidates']
    dist = arg(4, name=dist)                 ['dist']
    n_iters = arg(5, name=n_iters)           ['n_iters']
    delta = arg(6, name=delta)               ['delta']
    init_graph = arg(7, name=init_graph)     ['init_graph']
    rp_tree_init = arg(8, name=rp_tree_init) ['rp_tree_init']
    leaf_array = arg(9, name=leaf_array)     ['leaf_array']
    low_memory = arg(10, name=low_memory)    ['low_memory']
    verbose = arg(11, name=verbose)          ['verbose']
    $const6.1.1 = const(int, 0)              ['$const6.1.1']
    $8binary_subscr.2 = getitem(value=init_graph, index=$const6.1.1, fn=<built-in function getitem>) ['$8binary_subscr.2', '$const6.1.1', 'init_graph']
    $18load_attr.3 = getattr(value=$8binary_subscr.2, attr=shape) ['$18load_attr.3', '$8binary_subscr.2']
    $const28.4.1 = const(int, 0)             ['$const28.4.1']
    $30binary_subscr.5 = getitem(value=$18load_attr.3, index=$const28.4.1, fn=<built-in function getitem>) ['$18load_attr.3', '$30binary_subscr.5', '$const28.4.1']
    $const40.6.2 = const(int, 1)             ['$const40.6.2']
    $42compare_op.7 = $30binary_subscr.5 == $const40.6.2 ['$30binary_subscr.5', '$42compare_op.7', '$const40.6.2']
    bool48 = global(bool: <class 'bool'>)    ['bool48']
    $48pred = call bool48($42compare_op.7, func=bool48, args=(Var($42compare_op.7, pynndescent_.py:343),), kws=(), vararg=None, varkwarg=None, target=None) ['$42compare_op.7', '$48pred', 'bool48']
    branch $48pred, 50, 184                  ['$48pred']
label 50:
    $50load_global.0 = global(make_heap: CPUDispatcher(<function make_heap at 0x7fe136b2a200>)) ['$50load_global.0']
    $64load_attr.3 = getattr(value=data, attr=shape) ['$64load_attr.3', 'data']
    $const74.4.1 = const(int, 0)             ['$const74.4.1']
    $76binary_subscr.5 = getitem(value=$64load_attr.3, index=$const74.4.1, fn=<built-in function getitem>) ['$64load_attr.3', '$76binary_subscr.5', '$const74.4.1']
    current_graph = call $50load_global.0($76binary_subscr.5, n_neighbors, func=$50load_global.0, args=[Var($76binary_subscr.5, pynndescent_.py:344), Var(n_neighbors, pynndescent_.py:327)], kws=(), vararg=None, varkwarg=None, target=None) ['$50load_global.0', '$76binary_subscr.5', 'current_graph', 'n_neighbors']
    bool106 = global(bool: <class 'bool'>)   ['bool106']
    $106pred = call bool106(rp_tree_init, func=bool106, args=(Var(rp_tree_init, pynndescent_.py:327),), kws=(), vararg=None, varkwarg=None, target=None) ['$106pred', 'bool106', 'rp_tree_init']
    branch $106pred, 108, 144                ['$106pred']
label 108:
    $108load_global.0 = global(init_rp_tree: CPUDispatcher(<function init_rp_tree at 0x7fe136283240>)) ['$108load_global.0']
    $132call.6 = call $108load_global.0(data, dist, current_graph, leaf_array, func=$108load_global.0, args=[Var(data, pynndescent_.py:327), Var(dist, pynndescent_.py:327), Var(current_graph, pynndescent_.py:344), Var(leaf_array, pynndescent_.py:327)], kws=(), vararg=None, varkwarg=None, target=None) ['$108load_global.0', '$132call.6', 'current_graph', 'data', 'dist', 'leaf_array']
    jump 144                                 []
label 144:
    $144load_global.0 = global(init_random: CPUDispatcher(<function init_random at 0x7fe1367767a0>)) ['$144load_global.0']
    $170call.7 = call $144load_global.0(n_neighbors, data, current_graph, dist, rng_state, func=$144load_global.0, args=[Var(n_neighbors, pynndescent_.py:327), Var(data, pynndescent_.py:327), Var(current_graph, pynndescent_.py:344), Var(dist, pynndescent_.py:327), Var(rng_state, pynndescent_.py:327)], kws=(), vararg=None, varkwarg=None, target=None) ['$144load_global.0', '$170call.7', 'current_graph', 'data', 'dist', 'n_neighbors', 'rng_state']
    jump 334                                 []
label 184:
    $const186.1.1 = const(int, 0)            ['$const186.1.1']
    $188binary_subscr.2 = getitem(value=init_graph, index=$const186.1.1, fn=<built-in function getitem>) ['$188binary_subscr.2', '$const186.1.1', 'init_graph']
    $198load_attr.3 = getattr(value=$188binary_subscr.2, attr=shape) ['$188binary_subscr.2', '$198load_attr.3']
    $const208.4.1 = const(int, 0)            ['$const208.4.1']
    $210binary_subscr.5 = getitem(value=$198load_attr.3, index=$const208.4.1, fn=<built-in function getitem>) ['$198load_attr.3', '$210binary_subscr.5', '$const208.4.1']
    $222load_attr.7 = getattr(value=data, attr=shape) ['$222load_attr.7', 'data']
    $const232.8.1 = const(int, 0)            ['$const232.8.1']
    $234binary_subscr.9 = getitem(value=$222load_attr.7, index=$const232.8.1, fn=<built-in function getitem>) ['$222load_attr.7', '$234binary_subscr.9', '$const232.8.1']
    $244compare_op.10 = $210binary_subscr.5 == $234binary_subscr.9 ['$210binary_subscr.5', '$234binary_subscr.9', '$244compare_op.10']
    bool250 = global(bool: <class 'bool'>)   ['bool250']
    $250pred = call bool250($244compare_op.10, func=bool250, args=(Var($244compare_op.10, pynndescent_.py:351),), kws=(), vararg=None, varkwarg=None, target=None) ['$244compare_op.10', '$250pred', 'bool250']
    branch $250pred, 252, 304                ['$250pred']
label 252:
    $const254.1.1 = const(int, 0)            ['$const254.1.1']
    $256binary_subscr.2 = getitem(value=init_graph, index=$const254.1.1, fn=<built-in function getitem>) ['$256binary_subscr.2', '$const254.1.1', 'init_graph']
    $266load_attr.3 = getattr(value=$256binary_subscr.2, attr=shape) ['$256binary_subscr.2', '$266load_attr.3']
    $const276.4.2 = const(int, 1)            ['$const276.4.2']
    $278binary_subscr.5 = getitem(value=$266load_attr.3, index=$const276.4.2, fn=<built-in function getitem>) ['$266load_attr.3', '$278binary_subscr.5', '$const276.4.2']
    $290compare_op.7 = $278binary_subscr.5 == n_neighbors ['$278binary_subscr.5', '$290compare_op.7', 'n_neighbors']
    bool296 = global(bool: <class 'bool'>)   ['bool296']
    $296pred = call bool296($290compare_op.7, func=bool296, args=(Var($290compare_op.7, pynndescent_.py:352),), kws=(), vararg=None, varkwarg=None, target=None) ['$290compare_op.7', '$296pred', 'bool296']
    branch $296pred, 298, 304                ['$296pred']
label 298:
    current_graph = init_graph               ['current_graph', 'init_graph']
    jump 334                                 []
label 304:
    $304load_global.0 = global(ValueError: <class 'ValueError'>) ['$304load_global.0']
    $const316.2.3 = const(str, Invalid initial graph specified!) ['$const316.2.3']
    $322call.3 = call $304load_global.0($const316.2.3, func=$304load_global.0, args=[Var($const316.2.3, pynndescent_.py:356)], kws=(), vararg=None, varkwarg=None, target=None) ['$304load_global.0', '$322call.3', '$const316.2.3']
    raise $322call.3                         ['$322call.3']
label 334:
    bool336 = global(bool: <class 'bool'>)   ['bool336']
    $336pred = call bool336(low_memory, func=bool336, args=(Var(low_memory, pynndescent_.py:327),), kws=(), vararg=None, varkwarg=None, target=None) ['$336pred', 'bool336', 'low_memory']
    branch $336pred, 338, 388                ['$336pred']
label 338:
    $338load_global.0 = global(nn_descent_internal_low_memory_parallel: CPUDispatcher(<function nn_descent_internal_low_memory_parallel at 0x7fe136404860>)) ['$338load_global.0']
    $374call.11 = call $338load_global.0(current_graph, data, n_neighbors, rng_state, func=$338load_global.0, args=[Var(current_graph, pynndescent_.py:344), Var(data, pynndescent_.py:327), Var(n_neighbors, pynndescent_.py:327), Var(rng_state, pynndescent_.py:327)], kws=[('max_candidates', Var(max_candidates, pynndescent_.py:327)), ('dist', Var(dist, pynndescent_.py:327)), ('n_iters', Var(n_iters, pynndescent_.py:327)), ('delta', Var(delta, pynndescent_.py:327)), ('verbose', Var(verbose, pynndescent_.py:327))], vararg=None, varkwarg=None, target=None) ['$338load_global.0', '$374call.11', 'current_graph', 'data', 'delta', 'dist', 'max_candidates', 'n_iters', 'n_neighbors', 'rng_state', 'verbose']
    jump 436                                 []
label 388:
    $388load_global.0 = global(nn_descent_internal_high_memory_parallel: CPUDispatcher(<function nn_descent_internal_high_memory_parallel at 0x7fe1364051c0>)) ['$388load_global.0']
    $424call.11 = call $388load_global.0(current_graph, data, n_neighbors, rng_state, func=$388load_global.0, args=[Var(current_graph, pynndescent_.py:344), Var(data, pynndescent_.py:327), Var(n_neighbors, pynndescent_.py:327), Var(rng_state, pynndescent_.py:327)], kws=[('max_candidates', Var(max_candidates, pynndescent_.py:327)), ('dist', Var(dist, pynndescent_.py:327)), ('n_iters', Var(n_iters, pynndescent_.py:327)), ('delta', Var(delta, pynndescent_.py:327)), ('verbose', Var(verbose, pynndescent_.py:327))], vararg=None, varkwarg=None, target=None) ['$388load_global.0', '$424call.11', 'current_graph', 'data', 'delta', 'dist', 'max_candidates', 'n_iters', 'n_neighbors', 'rng_state', 'verbose']
    jump 436                                 []
label 436:
    $436load_global.0 = global(deheap_sort: CPUDispatcher(<function deheap_sort at 0x7fe136b29620>)) ['$436load_global.0']
    $const450.3.1 = const(int, 0)            ['$const450.3.1']
    $452binary_subscr.4 = getitem(value=current_graph, index=$const450.3.1, fn=<built-in function getitem>) ['$452binary_subscr.4', '$const450.3.1', 'current_graph']
    $const464.6.2 = const(int, 1)            ['$const464.6.2']
    $466binary_subscr.7 = getitem(value=current_graph, index=$const464.6.2, fn=<built-in function getitem>) ['$466binary_subscr.7', '$const464.6.2', 'current_graph']
    $480call.8 = call $436load_global.0($452binary_subscr.4, $466binary_subscr.7, func=$436load_global.0, args=[Var($452binary_subscr.4, pynndescent_.py:383), Var($466binary_subscr.7, pynndescent_.py:383)], kws=(), vararg=None, varkwarg=None, target=None) ['$436load_global.0', '$452binary_subscr.4', '$466binary_subscr.7', '$480call.8']
    $490return_value.9 = cast(value=$480call.8) ['$480call.8', '$490return_value.9']
    return $490return_value.9                ['$490return_value.9']
label 0:
    data = arg(0, name=data)                 ['data']
    dist = arg(1, name=dist)                 ['dist']
    current_graph = arg(2, name=current_graph) ['current_graph']
    leaf_array = arg(3, name=leaf_array)     ['leaf_array']
    $6load_attr.1 = getattr(value=leaf_array, attr=shape) ['$6load_attr.1', 'leaf_array']
    $const16.2.1 = const(int, 0)             ['$const16.2.1']
    n_leaves = getitem(value=$6load_attr.1, index=$const16.2.1, fn=<built-in function getitem>) ['$6load_attr.1', '$const16.2.1', 'n_leaves']
    block_size = const(int, 65536)           ['block_size']
    n_blocks = n_leaves // block_size        ['block_size', 'n_blocks', 'n_leaves']
    $44load_global.8 = global(range: <class 'range'>) ['$44load_global.8']
    $const58.11.3 = const(int, 1)            ['$const58.11.3']
    $binop_add60.12 = n_blocks + $const58.11.3 ['$binop_add60.12', '$const58.11.3', 'n_blocks']
    $68call.13 = call $44load_global.8($binop_add60.12, func=$44load_global.8, args=[Var($binop_add60.12, pynndescent_.py:104)], kws=(), vararg=None, varkwarg=None, target=None) ['$44load_global.8', '$68call.13', '$binop_add60.12']
    $78get_iter.14 = getiter(value=$68call.13) ['$68call.13', '$78get_iter.14']
    $phi82.0 = $78get_iter.14                ['$78get_iter.14', '$phi82.0']
    jump 80                                  []
label 80:
    jump 82                                  []
label 82:
    $82for_iter.1 = iternext(value=$phi82.0) ['$82for_iter.1', '$phi82.0']
    $82for_iter.2 = pair_first(value=$82for_iter.1) ['$82for_iter.1', '$82for_iter.2']
    $82for_iter.3 = pair_second(value=$82for_iter.1) ['$82for_iter.1', '$82for_iter.3']
    $phi84.1 = $82for_iter.2                 ['$82for_iter.2', '$phi84.1']
    branch $82for_iter.3, 84, 730            ['$82for_iter.3']
label 84:
    i = $phi84.1                             ['$phi84.1', 'i']
    block_start = i * block_size             ['block_size', 'block_start', 'i']
    $96load_global.5 = global(min: <built-in function min>) ['$96load_global.5']
    $const112.9.3 = const(int, 1)            ['$const112.9.3']
    $binop_add114.10 = i + $const112.9.3     ['$binop_add114.10', '$const112.9.3', 'i']
    $binop_mul120.12 = $binop_add114.10 * block_size ['$binop_add114.10', '$binop_mul120.12', 'block_size']
    block_end = call $96load_global.5(n_leaves, $binop_mul120.12, func=$96load_global.5, args=[Var(n_leaves, pynndescent_.py:100), Var($binop_mul120.12, pynndescent_.py:106)], kws=(), vararg=None, varkwarg=None, target=None) ['$96load_global.5', '$binop_mul120.12', 'block_end', 'n_leaves']
    $146build_slice.17 = global(slice: <class 'slice'>) ['$146build_slice.17']
    $146build_slice.18 = call $146build_slice.17(block_start, block_end, func=$146build_slice.17, args=(Var(block_start, pynndescent_.py:105), Var(block_end, pynndescent_.py:106)), kws=(), vararg=None, varkwarg=None, target=None) ['$146build_slice.17', '$146build_slice.18', 'block_end', 'block_start']
    leaf_block = getitem(value=leaf_array, index=$146build_slice.18, fn=<built-in function getitem>) ['$146build_slice.18', 'leaf_array', 'leaf_block']
    $const162.21.3 = const(int, 1)           ['$const162.21.3']
    $164binary_subscr.22 = getitem(value=current_graph, index=$const162.21.3, fn=<built-in function getitem>) ['$164binary_subscr.22', '$const162.21.3', 'current_graph']
    $const174.23.0 = const(NoneType, None)   ['$const174.23.0']
    $const176.24.0 = const(NoneType, None)   ['$const176.24.0']
    $178build_slice.25 = global(slice: <class 'slice'>) ['$178build_slice.25']
    $178build_slice.26 = call $178build_slice.25($const174.23.0, $const176.24.0, func=$178build_slice.25, args=(Var($const174.23.0, pynndescent_.py:109), Var($const176.24.0, pynndescent_.py:109)), kws=(), vararg=None, varkwarg=None, target=None) ['$178build_slice.25', '$178build_slice.26', '$const174.23.0', '$const176.24.0']
    $const180.27.1 = const(int, 0)           ['$const180.27.1']
    $182build_tuple.28 = build_tuple(items=[Var($178build_slice.26, pynndescent_.py:109), Var($const180.27.1, pynndescent_.py:109)]) ['$178build_slice.26', '$182build_tuple.28', '$const180.27.1']
    dist_thresholds = getitem(value=$164binary_subscr.22, index=$182build_tuple.28, fn=<built-in function getitem>) ['$164binary_subscr.22', '$182build_tuple.28', 'dist_thresholds']
    $196load_global.30 = global(generate_leaf_updates: CPUDispatcher(<function generate_leaf_updates at 0x7fe136815ee0>)) ['$196load_global.30']
    updates = call $196load_global.30(leaf_block, dist_thresholds, data, dist, func=$196load_global.30, args=[Var(leaf_block, pynndescent_.py:108), Var(dist_thresholds, pynndescent_.py:109), Var(data, pynndescent_.py:97), Var(dist, pynndescent_.py:97)], kws=(), vararg=None, varkwarg=None, target=None) ['$196load_global.30', 'data', 'dist', 'dist_thresholds', 'leaf_block', 'updates']
    $232load_global.37 = global(range: <class 'range'>) ['$232load_global.37']
    $244load_global.39 = global(len: <built-in function len>) ['$244load_global.39']
    $262call.42 = call $244load_global.39(updates, func=$244load_global.39, args=[Var(updates, pynndescent_.py:111)], kws=(), vararg=None, varkwarg=None, target=None) ['$244load_global.39', '$262call.42', 'updates']
    $276call.43 = call $232load_global.37($262call.42, func=$232load_global.37, args=[Var($262call.42, pynndescent_.py:113)], kws=(), vararg=None, varkwarg=None, target=None) ['$232load_global.37', '$262call.42', '$276call.43']
    $286get_iter.44 = getiter(value=$276call.43) ['$276call.43', '$286get_iter.44']
    $phi288.1 = $286get_iter.44              ['$286get_iter.44', '$phi288.1']
    jump 288                                 []
label 288:
    $288for_iter.2 = iternext(value=$phi288.1) ['$288for_iter.2', '$phi288.1']
    $288for_iter.3 = pair_first(value=$288for_iter.2) ['$288for_iter.2', '$288for_iter.3']
    $288for_iter.4 = pair_second(value=$288for_iter.2) ['$288for_iter.2', '$288for_iter.4']
    $phi290.2 = $288for_iter.3               ['$288for_iter.3', '$phi290.2']
    branch $288for_iter.4, 290, 726          ['$288for_iter.4']
label 290:
    j = $phi290.2                            ['$phi290.2', 'j']
    $292load_global.3 = global(range: <class 'range'>) ['$292load_global.3']
    $304load_global.5 = global(len: <built-in function len>) ['$304load_global.5']
    $320binary_subscr.9 = getitem(value=updates, index=j, fn=<built-in function getitem>) ['$320binary_subscr.9', 'j', 'updates']
    $334call.10 = call $304load_global.5($320binary_subscr.9, func=$304load_global.5, args=[Var($320binary_subscr.9, pynndescent_.py:114)], kws=(), vararg=None, varkwarg=None, target=None) ['$304load_global.5', '$320binary_subscr.9', '$334call.10']
    $348call.11 = call $292load_global.3($334call.10, func=$292load_global.3, args=[Var($334call.10, pynndescent_.py:114)], kws=(), vararg=None, varkwarg=None, target=None) ['$292load_global.3', '$334call.10', '$348call.11']
    $358get_iter.12 = getiter(value=$348call.11) ['$348call.11', '$358get_iter.12']
    $phi360.2 = $358get_iter.12              ['$358get_iter.12', '$phi360.2']
    jump 360                                 []
label 360:
    $360for_iter.3 = iternext(value=$phi360.2) ['$360for_iter.3', '$phi360.2']
    $360for_iter.4 = pair_first(value=$360for_iter.3) ['$360for_iter.3', '$360for_iter.4']
    $360for_iter.5 = pair_second(value=$360for_iter.3) ['$360for_iter.3', '$360for_iter.5']
    $phi362.3 = $360for_iter.4               ['$360for_iter.4', '$phi362.3']
    branch $360for_iter.5, 362, 724          ['$360for_iter.5']
label 362:
    k = $phi362.3                            ['$phi362.3', 'k']
    $368binary_subscr.6 = getitem(value=updates, index=j, fn=<built-in function getitem>) ['$368binary_subscr.6', 'j', 'updates']
    $380binary_subscr.8 = getitem(value=$368binary_subscr.6, index=k, fn=<built-in function getitem>) ['$368binary_subscr.6', '$380binary_subscr.8', 'k']
    $390unpack_sequence.12 = exhaust_iter(value=$380binary_subscr.8, count=3) ['$380binary_subscr.8', '$390unpack_sequence.12']
    $390unpack_sequence.9 = static_getitem(value=$390unpack_sequence.12, index=0, index_var=None, fn=<built-in function getitem>) ['$390unpack_sequence.12', '$390unpack_sequence.9']
    $390unpack_sequence.10 = static_getitem(value=$390unpack_sequence.12, index=1, index_var=None, fn=<built-in function getitem>) ['$390unpack_sequence.10', '$390unpack_sequence.12']
    $390unpack_sequence.11 = static_getitem(value=$390unpack_sequence.12, index=2, index_var=None, fn=<built-in function getitem>) ['$390unpack_sequence.11', '$390unpack_sequence.12']
    p = $390unpack_sequence.9                ['$390unpack_sequence.9', 'p']
    q = $390unpack_sequence.10               ['$390unpack_sequence.10', 'q']
    d = $390unpack_sequence.11               ['$390unpack_sequence.11', 'd']
    $const402.14.4 = const(int, -1)          ['$const402.14.4']
    $404compare_op.15 = p == $const402.14.4  ['$404compare_op.15', '$const402.14.4', 'p']
    bool410 = global(bool: <class 'bool'>)   ['bool410']
    $410pred = call bool410($404compare_op.15, func=bool410, args=(Var($404compare_op.15, pynndescent_.py:117),), kws=(), vararg=None, varkwarg=None, target=None) ['$404compare_op.15', '$410pred', 'bool410']
    branch $410pred, 424, 412                ['$410pred']
label 412:
    $const414.4.4 = const(int, -1)           ['$const414.4.4']
    $416compare_op.5 = q == $const414.4.4    ['$416compare_op.5', '$const414.4.4', 'q']
    bool422 = global(bool: <class 'bool'>)   ['bool422']
    $422pred = call bool422($416compare_op.5, func=bool422, args=(Var($416compare_op.5, pynndescent_.py:117),), kws=(), vararg=None, varkwarg=None, target=None) ['$416compare_op.5', '$422pred', 'bool422']
    branch $422pred, 424, 426                ['$422pred']
label 424:
    jump 360                                 []
label 426:
    $426load_global.3 = global(checked_flagged_heap_push: CPUDispatcher(<function checked_flagged_heap_push at 0x7fe136b2b420>)) ['$426load_global.3']
    $const440.6.3 = const(int, 1)            ['$const440.6.3']
    $442binary_subscr.7 = getitem(value=current_graph, index=$const440.6.3, fn=<built-in function getitem>) ['$442binary_subscr.7', '$const440.6.3', 'current_graph']
    $454binary_subscr.9 = getitem(value=$442binary_subscr.7, index=p, fn=<built-in function getitem>) ['$442binary_subscr.7', '$454binary_subscr.9', 'p']
    $const466.11.1 = const(int, 0)           ['$const466.11.1']
    $468binary_subscr.12 = getitem(value=current_graph, index=$const466.11.1, fn=<built-in function getitem>) ['$468binary_subscr.12', '$const466.11.1', 'current_graph']
    $480binary_subscr.14 = getitem(value=$468binary_subscr.12, index=p, fn=<built-in function getitem>) ['$468binary_subscr.12', '$480binary_subscr.14', 'p']
    $const492.16.5 = const(int, 2)           ['$const492.16.5']
    $494binary_subscr.17 = getitem(value=current_graph, index=$const492.16.5, fn=<built-in function getitem>) ['$494binary_subscr.17', '$const492.16.5', 'current_graph']
    $506binary_subscr.19 = getitem(value=$494binary_subscr.17, index=p, fn=<built-in function getitem>) ['$494binary_subscr.17', '$506binary_subscr.19', 'p']
    $520load_global.22 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$520load_global.22']
    $532load_attr.24 = getattr(value=$520load_global.22, attr=uint8) ['$520load_global.22', '$532load_attr.24']
    $const542.25.3 = const(int, 1)           ['$const542.25.3']
    $548call.26 = call $532load_attr.24($const542.25.3, func=$532load_attr.24, args=[Var($const542.25.3, pynndescent_.py:126)], kws=(), vararg=None, varkwarg=None, target=None) ['$532load_attr.24', '$548call.26', '$const542.25.3']
    $562call.27 = call $426load_global.3($454binary_subscr.9, $480binary_subscr.14, $506binary_subscr.19, d, q, $548call.26, func=$426load_global.3, args=[Var($454binary_subscr.9, pynndescent_.py:121), Var($480binary_subscr.14, pynndescent_.py:122), Var($506binary_subscr.19, pynndescent_.py:123), Var(d, pynndescent_.py:115), Var(q, pynndescent_.py:115), Var($548call.26, pynndescent_.py:126)], kws=(), vararg=None, varkwarg=None, target=None) ['$426load_global.3', '$454binary_subscr.9', '$480binary_subscr.14', '$506binary_subscr.19', '$548call.26', '$562call.27', 'd', 'q']
    $574load_global.28 = global(checked_flagged_heap_push: CPUDispatcher(<function checked_flagged_heap_push at 0x7fe136b2b420>)) ['$574load_global.28']
    $const588.31.3 = const(int, 1)           ['$const588.31.3']
    $590binary_subscr.32 = getitem(value=current_graph, index=$const588.31.3, fn=<built-in function getitem>) ['$590binary_subscr.32', '$const588.31.3', 'current_graph']
    $602binary_subscr.34 = getitem(value=$590binary_subscr.32, index=q, fn=<built-in function getitem>) ['$590binary_subscr.32', '$602binary_subscr.34', 'q']
    $const614.36.1 = const(int, 0)           ['$const614.36.1']
    $616binary_subscr.37 = getitem(value=current_graph, index=$const614.36.1, fn=<built-in function getitem>) ['$616binary_subscr.37', '$const614.36.1', 'current_graph']
    $628binary_subscr.39 = getitem(value=$616binary_subscr.37, index=q, fn=<built-in function getitem>) ['$616binary_subscr.37', '$628binary_subscr.39', 'q']
    $const640.41.5 = const(int, 2)           ['$const640.41.5']
    $642binary_subscr.42 = getitem(value=current_graph, index=$const640.41.5, fn=<built-in function getitem>) ['$642binary_subscr.42', '$const640.41.5', 'current_graph']
    $654binary_subscr.44 = getitem(value=$642binary_subscr.42, index=q, fn=<built-in function getitem>) ['$642binary_subscr.42', '$654binary_subscr.44', 'q']
    $668load_global.47 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$668load_global.47']
    $680load_attr.49 = getattr(value=$668load_global.47, attr=uint8) ['$668load_global.47', '$680load_attr.49']
    $const690.50.3 = const(int, 1)           ['$const690.50.3']
    $696call.51 = call $680load_attr.49($const690.50.3, func=$680load_attr.49, args=[Var($const690.50.3, pynndescent_.py:134)], kws=(), vararg=None, varkwarg=None, target=None) ['$680load_attr.49', '$696call.51', '$const690.50.3']
    $710call.52 = call $574load_global.28($602binary_subscr.34, $628binary_subscr.39, $654binary_subscr.44, d, p, $696call.51, func=$574load_global.28, args=[Var($602binary_subscr.34, pynndescent_.py:129), Var($628binary_subscr.39, pynndescent_.py:130), Var($654binary_subscr.44, pynndescent_.py:131), Var(d, pynndescent_.py:115), Var(p, pynndescent_.py:115), Var($696call.51, pynndescent_.py:134)], kws=(), vararg=None, varkwarg=None, target=None) ['$574load_global.28', '$602binary_subscr.34', '$628binary_subscr.39', '$654binary_subscr.44', '$696call.51', '$710call.52', 'd', 'p']
    jump 360                                 []
label 724:
    jump 288                                 []
label 726:
    jump 80                                  []
label 730:
    $const730.0.0 = const(NoneType, None)    ['$const730.0.0']
    $732return_value.1 = cast(value=$const730.0.0) ['$732return_value.1', '$const730.0.0']
    return $732return_value.1                ['$732return_value.1']
label 0:
    leaf_block = arg(0, name=leaf_block)     ['leaf_block']
    dist_thresholds = arg(1, name=dist_thresholds) ['dist_thresholds']
    data = arg(2, name=data)                 ['data']
    dist = arg(3, name=dist)                 ['dist']
    $const4.0.1 = const(code, <code object <listcomp> at 0x7fe136c53d20, file "/root/subliminal-learning/.venv/lib/python3.11/site-packages/pynndescent/pynndescent_.py", line 77>) ['$const4.0.1']
    $6make_function.1 = make_function(name=None, code=<code object <listcomp> at 0x7fe136c53d20, file "/root/subliminal-learning/.venv/lib/python3.11/site-packages/pynndescent/pynndescent_.py", line 77>, closure=None, defaults=None) ['$6make_function.1']
    $8load_global.2 = global(range: <class 'range'>) ['$8load_global.2']
    $22load_attr.5 = getattr(value=leaf_block, attr=shape) ['$22load_attr.5', 'leaf_block']
    $const32.6.2 = const(int, 0)             ['$const32.6.2']
    $34binary_subscr.7 = getitem(value=$22load_attr.5, index=$const32.6.2, fn=<built-in function getitem>) ['$22load_attr.5', '$34binary_subscr.7', '$const32.6.2']
    $48call.8 = call $8load_global.2($34binary_subscr.7, func=$8load_global.2, args=[Var($34binary_subscr.7, pynndescent_.py:77)], kws=(), vararg=None, varkwarg=None, target=None) ['$34binary_subscr.7', '$48call.8', '$8load_global.2']
    $58get_iter.9 = getiter(value=$48call.8) ['$48call.8', '$58get_iter.9']
    updates = call $6make_function.1($58get_iter.9, func=$6make_function.1, args=[Var($58get_iter.9, pynndescent_.py:77)], kws=(), vararg=None, varkwarg=None, target=None) ['$58get_iter.9', '$6make_function.1', 'updates']
    $76load_global.11 = global(numba: <module 'numba' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/__init__.py'>) ['$76load_global.11']
    $88load_attr.13 = getattr(value=$76load_global.11, attr=prange) ['$76load_global.11', '$88load_attr.13']
    $100load_attr.15 = getattr(value=leaf_block, attr=shape) ['$100load_attr.15', 'leaf_block']
    $const110.16.2 = const(int, 0)           ['$const110.16.2']
    $112binary_subscr.17 = getitem(value=$100load_attr.15, index=$const110.16.2, fn=<built-in function getitem>) ['$100load_attr.15', '$112binary_subscr.17', '$const110.16.2']
    $126call.18 = call $88load_attr.13($112binary_subscr.17, func=$88load_attr.13, args=[Var($112binary_subscr.17, pynndescent_.py:79)], kws=(), vararg=None, varkwarg=None, target=None) ['$112binary_subscr.17', '$126call.18', '$88load_attr.13']
    $136get_iter.19 = getiter(value=$126call.18) ['$126call.18', '$136get_iter.19']
    $phi138.0 = $136get_iter.19              ['$136get_iter.19', '$phi138.0']
    jump 138                                 []
label 138:
    $138for_iter.1 = iternext(value=$phi138.0) ['$138for_iter.1', '$phi138.0']
    $138for_iter.2 = pair_first(value=$138for_iter.1) ['$138for_iter.1', '$138for_iter.2']
    $138for_iter.3 = pair_second(value=$138for_iter.1) ['$138for_iter.1', '$138for_iter.3']
    $phi140.1 = $138for_iter.2               ['$138for_iter.2', '$phi140.1']
    branch $138for_iter.3, 140, 496          ['$138for_iter.3']
label 140:
    n = $phi140.1                            ['$phi140.1', 'n']
    $142load_global.2 = global(range: <class 'range'>) ['$142load_global.2']
    $156load_attr.5 = getattr(value=leaf_block, attr=shape) ['$156load_attr.5', 'leaf_block']
    $const166.6.3 = const(int, 1)            ['$const166.6.3']
    $168binary_subscr.7 = getitem(value=$156load_attr.5, index=$const166.6.3, fn=<built-in function getitem>) ['$156load_attr.5', '$168binary_subscr.7', '$const166.6.3']
    $182call.8 = call $142load_global.2($168binary_subscr.7, func=$142load_global.2, args=[Var($168binary_subscr.7, pynndescent_.py:80)], kws=(), vararg=None, varkwarg=None, target=None) ['$142load_global.2', '$168binary_subscr.7', '$182call.8']
    $192get_iter.9 = getiter(value=$182call.8) ['$182call.8', '$192get_iter.9']
    $phi194.1 = $192get_iter.9               ['$192get_iter.9', '$phi194.1']
    jump 194                                 []
label 194:
    $194for_iter.2 = iternext(value=$phi194.1) ['$194for_iter.2', '$phi194.1']
    $194for_iter.3 = pair_first(value=$194for_iter.2) ['$194for_iter.2', '$194for_iter.3']
    $194for_iter.4 = pair_second(value=$194for_iter.2) ['$194for_iter.2', '$194for_iter.4']
    $phi196.2 = $194for_iter.3               ['$194for_iter.3', '$phi196.2']
    branch $194for_iter.4, 196, 494          ['$194for_iter.4']
label 196:
    i = $phi196.2                            ['$phi196.2', 'i']
    $204build_tuple.6 = build_tuple(items=[Var(n, pynndescent_.py:79), Var(i, pynndescent_.py:80)]) ['$204build_tuple.6', 'i', 'n']
    p = getitem(value=leaf_block, index=$204build_tuple.6, fn=<built-in function getitem>) ['$204build_tuple.6', 'leaf_block', 'p']
    $const220.9.2 = const(int, 0)            ['$const220.9.2']
    $222compare_op.10 = p < $const220.9.2    ['$222compare_op.10', '$const220.9.2', 'p']
    bool228 = global(bool: <class 'bool'>)   ['bool228']
    $228pred = call bool228($222compare_op.10, func=bool228, args=(Var($222compare_op.10, pynndescent_.py:82),), kws=(), vararg=None, varkwarg=None, target=None) ['$222compare_op.10', '$228pred', 'bool228']
    branch $228pred, 230, 234                ['$228pred']
label 230:
    jump 494                                 []
label 234:
    $234load_global.2 = global(range: <class 'range'>) ['$234load_global.2']
    $const248.5.3 = const(int, 1)            ['$const248.5.3']
    $binop_add250.6 = i + $const248.5.3      ['$binop_add250.6', '$const248.5.3', 'i']
    $256load_attr.8 = getattr(value=leaf_block, attr=shape) ['$256load_attr.8', 'leaf_block']
    $const266.9.3 = const(int, 1)            ['$const266.9.3']
    $268binary_subscr.10 = getitem(value=$256load_attr.8, index=$const266.9.3, fn=<built-in function getitem>) ['$256load_attr.8', '$268binary_subscr.10', '$const266.9.3']
    $282call.11 = call $234load_global.2($binop_add250.6, $268binary_subscr.10, func=$234load_global.2, args=[Var($binop_add250.6, pynndescent_.py:85), Var($268binary_subscr.10, pynndescent_.py:85)], kws=(), vararg=None, varkwarg=None, target=None) ['$234load_global.2', '$268binary_subscr.10', '$282call.11', '$binop_add250.6']
    $292get_iter.12 = getiter(value=$282call.11) ['$282call.11', '$292get_iter.12']
    $phi294.2 = $292get_iter.12              ['$292get_iter.12', '$phi294.2']
    jump 294                                 []
label 294:
    $294for_iter.3 = iternext(value=$phi294.2) ['$294for_iter.3', '$phi294.2']
    $294for_iter.4 = pair_first(value=$294for_iter.3) ['$294for_iter.3', '$294for_iter.4']
    $294for_iter.5 = pair_second(value=$294for_iter.3) ['$294for_iter.3', '$294for_iter.5']
    $phi296.3 = $294for_iter.4               ['$294for_iter.4', '$phi296.3']
    branch $294for_iter.5, 296, 492          ['$294for_iter.5']
label 296:
    j = $phi296.3                            ['$phi296.3', 'j']
    $304build_tuple.7 = build_tuple(items=[Var(n, pynndescent_.py:79), Var(j, pynndescent_.py:85)]) ['$304build_tuple.7', 'j', 'n']
    q = getitem(value=leaf_block, index=$304build_tuple.7, fn=<built-in function getitem>) ['$304build_tuple.7', 'leaf_block', 'q']
    $const320.10.2 = const(int, 0)           ['$const320.10.2']
    $322compare_op.11 = q < $const320.10.2   ['$322compare_op.11', '$const320.10.2', 'q']
    bool328 = global(bool: <class 'bool'>)   ['bool328']
    $328pred = call bool328($322compare_op.11, func=bool328, args=(Var($322compare_op.11, pynndescent_.py:87),), kws=(), vararg=None, varkwarg=None, target=None) ['$322compare_op.11', '$328pred', 'bool328']
    branch $328pred, 330, 334                ['$328pred']
label 330:
    jump 492                                 []
label 334:
    $342binary_subscr.7 = getitem(value=data, index=p, fn=<built-in function getitem>) ['$342binary_subscr.7', 'data', 'p']
    $356binary_subscr.10 = getitem(value=data, index=q, fn=<built-in function getitem>) ['$356binary_subscr.10', 'data', 'q']
    d = call dist($342binary_subscr.7, $356binary_subscr.10, func=dist, args=[Var($342binary_subscr.7, pynndescent_.py:90), Var($356binary_subscr.10, pynndescent_.py:90)], kws=(), vararg=None, varkwarg=None, target=None) ['$342binary_subscr.7', '$356binary_subscr.10', 'd', 'dist']
    $388binary_subscr.15 = getitem(value=dist_thresholds, index=p, fn=<built-in function getitem>) ['$388binary_subscr.15', 'dist_thresholds', 'p']
    $398compare_op.16 = d < $388binary_subscr.15 ['$388binary_subscr.15', '$398compare_op.16', 'd']
    bool404 = global(bool: <class 'bool'>)   ['bool404']
    $404pred = call bool404($398compare_op.16, func=bool404, args=(Var($398compare_op.16, pynndescent_.py:91),), kws=(), vararg=None, varkwarg=None, target=None) ['$398compare_op.16', '$404pred', 'bool404']
    branch $404pred, 430, 406                ['$404pred']
label 406:
    $412binary_subscr.6 = getitem(value=dist_thresholds, index=q, fn=<built-in function getitem>) ['$412binary_subscr.6', 'dist_thresholds', 'q']
    $422compare_op.7 = d < $412binary_subscr.6 ['$412binary_subscr.6', '$422compare_op.7', 'd']
    bool428 = global(bool: <class 'bool'>)   ['bool428']
    $428pred = call bool428($422compare_op.7, func=bool428, args=(Var($422compare_op.7, pynndescent_.py:91),), kws=(), vararg=None, varkwarg=None, target=None) ['$422compare_op.7', '$428pred', 'bool428']
    branch $428pred, 430, 490                ['$428pred']
label 430:
    $434binary_subscr.5 = getitem(value=updates, index=n, fn=<built-in function getitem>) ['$434binary_subscr.5', 'n', 'updates']
    $444load_method.7 = getattr(value=$434binary_subscr.5, attr=append) ['$434binary_subscr.5', '$444load_method.7']
    $472build_tuple.11 = build_tuple(items=[Var(p, pynndescent_.py:81), Var(q, pynndescent_.py:86), Var(d, pynndescent_.py:90)]) ['$472build_tuple.11', 'd', 'p', 'q']
    $478call.12 = call $444load_method.7($472build_tuple.11, func=$444load_method.7, args=[Var($472build_tuple.11, pynndescent_.py:92)], kws=(), vararg=None, varkwarg=None, target=None) ['$444load_method.7', '$472build_tuple.11', '$478call.12']
    jump 490                                 []
label 490:
    jump 294                                 []
label 492:
    jump 194                                 []
label 494:
    jump 138                                 []
label 496:
    $498return_value.1 = cast(value=updates) ['$498return_value.1', 'updates']
    return $498return_value.1                ['$498return_value.1']
label 0:
    implicit0 = arg(0, name=implicit0)       ['implicit0']
    $4build_list.0 = build_list(items=[])    ['$4build_list.0']
    $phi8.1 = implicit0                      ['$phi8.1', 'implicit0']
    $phi10.0 = $4build_list.0                ['$4build_list.0', '$phi10.0']
    $phi46.0 = $4build_list.0                ['$4build_list.0', '$phi46.0']
    jump 8                                   []
label 8:
    $8for_iter.2 = iternext(value=$phi8.1)   ['$8for_iter.2', '$phi8.1']
    $8for_iter.3 = pair_first(value=$8for_iter.2) ['$8for_iter.2', '$8for_iter.3']
    $8for_iter.4 = pair_second(value=$8for_iter.2) ['$8for_iter.2', '$8for_iter.4']
    $phi10.2 = $8for_iter.3                  ['$8for_iter.3', '$phi10.2']
    branch $8for_iter.4, 10, 46              ['$8for_iter.4']
label 10:
    i = $phi10.2                             ['$phi10.2', 'i']
    $const12.3.0 = const(int, -1)            ['$const12.3.0']
    $const14.4.0 = const(int, -1)            ['$const14.4.0']
    $16load_global.5 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$16load_global.5']
    $28load_attr.6 = getattr(value=$16load_global.5, attr=inf) ['$16load_global.5', '$28load_attr.6']
    $38build_tuple.7 = build_tuple(items=[Var($const12.3.0, pynndescent_.py:77), Var($const14.4.0, pynndescent_.py:77), Var($28load_attr.6, pynndescent_.py:77)]) ['$28load_attr.6', '$38build_tuple.7', '$const12.3.0', '$const14.4.0']
    $40build_list.8 = build_list(items=[Var($38build_tuple.7, pynndescent_.py:77)]) ['$38build_tuple.7', '$40build_list.8']
    $42list_append.9 = getattr(value=$phi10.0, attr=append) ['$42list_append.9', '$phi10.0']
    $42list_append.10 = call $42list_append.9($40build_list.8, func=$42list_append.9, args=(Var($40build_list.8, pynndescent_.py:77),), kws=(), vararg=None, varkwarg=None, target=None) ['$40build_list.8', '$42list_append.10', '$42list_append.9']
    jump 8                                   []
label 46:
    $46return_value.1 = cast(value=$phi46.0) ['$46return_value.1', '$phi46.0']
    return $46return_value.1                 ['$46return_value.1']
label 0:
    sched = arg(0, name=sched)               ['sched']
    closure____locals______listcomp____v169____v4build__list__0 = arg(1, name=closure____locals______listcomp____v169____v4build__list__0) ['closure____locals______listcomp____v169____v4build__list__0']
    data = arg(2, name=data)                 ['data']
    dist = arg(3, name=dist)                 ['dist']
    dist__thresholds = arg(4, name=dist__thresholds) ['dist__thresholds']
    leaf__block = arg(5, name=leaf__block)   ['leaf__block']
    $4load_global.0 = global(range: <class 'range'>) ['$4load_global.0']
    $const18.3.1 = const(int, 0)             ['$const18.3.1']
    $20binary_subscr.4 = getitem(value=sched, index=$const18.3.1, fn=<built-in function getitem>) ['$20binary_subscr.4', '$const18.3.1', 'sched']
    $const32.6.2 = const(int, 1)             ['$const32.6.2']
    $34binary_subscr.7 = getitem(value=sched, index=$const32.6.2, fn=<built-in function getitem>) ['$34binary_subscr.7', '$const32.6.2', 'sched']
    $44load_global.8 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$44load_global.8']
    $56load_method.10 = getattr(value=$44load_global.8, attr=uint8) ['$44load_global.8', '$56load_method.10']
    $const78.11.2 = const(int, 1)            ['$const78.11.2']
    $84call.12 = call $56load_method.10($const78.11.2, func=$56load_method.10, args=[Var($const78.11.2, <string>:2)], kws=(), vararg=None, varkwarg=None, target=None) ['$56load_method.10', '$84call.12', '$const78.11.2']
    $binop_add94.13 = $34binary_subscr.7 + $84call.12 ['$34binary_subscr.7', '$84call.12', '$binop_add94.13']
    $102call.14 = call $4load_global.0($20binary_subscr.4, $binop_add94.13, func=$4load_global.0, args=[Var($20binary_subscr.4, <string>:2), Var($binop_add94.13, <string>:2)], kws=(), vararg=None, varkwarg=None, target=None) ['$102call.14', '$20binary_subscr.4', '$4load_global.0', '$binop_add94.13']
    $112get_iter.15 = getiter(value=$102call.14) ['$102call.14', '$112get_iter.15']
    $phi114.0 = $112get_iter.15              ['$112get_iter.15', '$phi114.0']
    jump 114                                 []
label 114:
    $114for_iter.1 = iternext(value=$phi114.0) ['$114for_iter.1', '$phi114.0']
    $114for_iter.2 = pair_first(value=$114for_iter.1) ['$114for_iter.1', '$114for_iter.2']
    $114for_iter.3 = pair_second(value=$114for_iter.1) ['$114for_iter.1', '$114for_iter.3']
    $phi116.1 = $114for_iter.2               ['$114for_iter.2', '$phi116.1']
    branch $114for_iter.3, 116, 124          ['$114for_iter.3']
label 116:
    parfor__index_18 = $phi116.1             ['$phi116.1', 'parfor__index_18']
    __sentinel__0 = const(int, 0)            ['__sentinel__0']
    jump 114                                 []
label 124:
    $const124.0.0 = const(NoneType, None)    ['$const124.0.0']
    $126return_value.1 = cast(value=$const124.0.0) ['$126return_value.1', '$const124.0.0']
    return $126return_value.1                ['$126return_value.1']
label 0:
    n_neighbors = arg(0, name=n_neighbors)   ['n_neighbors']
    data = arg(1, name=data)                 ['data']
    heap = arg(2, name=heap)                 ['heap']
    dist = arg(3, name=dist)                 ['dist']
    rng_state = arg(4, name=rng_state)       ['rng_state']
    $4load_global.0 = global(range: <class 'range'>) ['$4load_global.0']
    $18load_attr.3 = getattr(value=data, attr=shape) ['$18load_attr.3', 'data']
    $const28.4.1 = const(int, 0)             ['$const28.4.1']
    $30binary_subscr.5 = getitem(value=$18load_attr.3, index=$const28.4.1, fn=<built-in function getitem>) ['$18load_attr.3', '$30binary_subscr.5', '$const28.4.1']
    $44call.6 = call $4load_global.0($30binary_subscr.5, func=$4load_global.0, args=[Var($30binary_subscr.5, pynndescent_.py:144)], kws=(), vararg=None, varkwarg=None, target=None) ['$30binary_subscr.5', '$44call.6', '$4load_global.0']
    $54get_iter.7 = getiter(value=$44call.6) ['$44call.6', '$54get_iter.7']
    $phi56.0 = $54get_iter.7                 ['$54get_iter.7', '$phi56.0']
    jump 56                                  []
label 56:
    $56for_iter.1 = iternext(value=$phi56.0) ['$56for_iter.1', '$phi56.0']
    $56for_iter.2 = pair_first(value=$56for_iter.1) ['$56for_iter.1', '$56for_iter.2']
    $56for_iter.3 = pair_second(value=$56for_iter.1) ['$56for_iter.1', '$56for_iter.3']
    $phi58.1 = $56for_iter.2                 ['$56for_iter.2', '$phi58.1']
    branch $56for_iter.3, 58, 502            ['$56for_iter.3']
label 58:
    i = $phi58.1                             ['$phi58.1', 'i']
    $const62.3.1 = const(int, 0)             ['$const62.3.1']
    $64binary_subscr.4 = getitem(value=heap, index=$const62.3.1, fn=<built-in function getitem>) ['$64binary_subscr.4', '$const62.3.1', 'heap']
    $const76.6.1 = const(int, 0)             ['$const76.6.1']
    $78build_tuple.7 = build_tuple(items=[Var(i, pynndescent_.py:144), Var($const76.6.1, pynndescent_.py:145)]) ['$78build_tuple.7', '$const76.6.1', 'i']
    $80binary_subscr.8 = getitem(value=$64binary_subscr.4, index=$78build_tuple.7, fn=<built-in function getitem>) ['$64binary_subscr.4', '$78build_tuple.7', '$80binary_subscr.8']
    $const90.9.2 = const(float, 0.0)         ['$const90.9.2']
    $92compare_op.10 = $80binary_subscr.8 < $const90.9.2 ['$80binary_subscr.8', '$92compare_op.10', '$const90.9.2']
    bool98 = global(bool: <class 'bool'>)    ['bool98']
    $98pred = call bool98($92compare_op.10, func=bool98, args=(Var($92compare_op.10, pynndescent_.py:145),), kws=(), vararg=None, varkwarg=None, target=None) ['$92compare_op.10', '$98pred', 'bool98']
    branch $98pred, 100, 500                 ['$98pred']
label 100:
    $100load_global.1 = global(range: <class 'range'>) ['$100load_global.1']
    $114load_global.4 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$114load_global.4']
    $126load_attr.6 = getattr(value=$114load_global.4, attr=sum) ['$114load_global.4', '$126load_attr.6']
    $const138.8.1 = const(int, 0)            ['$const138.8.1']
    $140binary_subscr.9 = getitem(value=heap, index=$const138.8.1, fn=<built-in function getitem>) ['$140binary_subscr.9', '$const138.8.1', 'heap']
    $152binary_subscr.11 = getitem(value=$140binary_subscr.9, index=i, fn=<built-in function getitem>) ['$140binary_subscr.9', '$152binary_subscr.11', 'i']
    $const162.12.2 = const(float, 0.0)       ['$const162.12.2']
    $164compare_op.13 = $152binary_subscr.11 >= $const162.12.2 ['$152binary_subscr.11', '$164compare_op.13', '$const162.12.2']
    $174call.14 = call $126load_attr.6($164compare_op.13, func=$126load_attr.6, args=[Var($164compare_op.13, pynndescent_.py:146)], kws=(), vararg=None, varkwarg=None, target=None) ['$126load_attr.6', '$164compare_op.13', '$174call.14']
    $binop_sub184.15 = n_neighbors - $174call.14 ['$174call.14', '$binop_sub184.15', 'n_neighbors']
    $192call.16 = call $100load_global.1($binop_sub184.15, func=$100load_global.1, args=[Var($binop_sub184.15, pynndescent_.py:146)], kws=(), vararg=None, varkwarg=None, target=None) ['$100load_global.1', '$192call.16', '$binop_sub184.15']
    $202get_iter.17 = getiter(value=$192call.16) ['$192call.16', '$202get_iter.17']
    $phi204.1 = $202get_iter.17              ['$202get_iter.17', '$phi204.1']
    jump 204                                 []
label 204:
    $204for_iter.2 = iternext(value=$phi204.1) ['$204for_iter.2', '$phi204.1']
    $204for_iter.3 = pair_first(value=$204for_iter.2) ['$204for_iter.2', '$204for_iter.3']
    $204for_iter.4 = pair_second(value=$204for_iter.2) ['$204for_iter.2', '$204for_iter.4']
    $phi206.2 = $204for_iter.3               ['$204for_iter.3', '$phi206.2']
    branch $204for_iter.4, 206, 500          ['$204for_iter.4']
label 206:
    j = $phi206.2                            ['$phi206.2', 'j']
    $208load_global.3 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$208load_global.3']
    $220load_attr.5 = getattr(value=$208load_global.3, attr=abs) ['$208load_global.3', '$220load_attr.5']
    $230load_global.6 = global(tau_rand_int: CPUDispatcher(<function tau_rand_int at 0x7fe136c69a80>)) ['$230load_global.6']
    $248call.9 = call $230load_global.6(rng_state, func=$230load_global.6, args=[Var(rng_state, pynndescent_.py:138)], kws=(), vararg=None, varkwarg=None, target=None) ['$230load_global.6', '$248call.9', 'rng_state']
    $262call.10 = call $220load_attr.5($248call.9, func=$220load_attr.5, args=[Var($248call.9, pynndescent_.py:147)], kws=(), vararg=None, varkwarg=None, target=None) ['$220load_attr.5', '$248call.9', '$262call.10']
    $274load_attr.12 = getattr(value=data, attr=shape) ['$274load_attr.12', 'data']
    $const284.13.1 = const(int, 0)           ['$const284.13.1']
    $286binary_subscr.14 = getitem(value=$274load_attr.12, index=$const284.13.1, fn=<built-in function getitem>) ['$274load_attr.12', '$286binary_subscr.14', '$const284.13.1']
    idx = $262call.10 % $286binary_subscr.14 ['$262call.10', '$286binary_subscr.14', 'idx']
    $310binary_subscr.20 = getitem(value=data, index=idx, fn=<built-in function getitem>) ['$310binary_subscr.20', 'data', 'idx']
    $324binary_subscr.23 = getitem(value=data, index=i, fn=<built-in function getitem>) ['$324binary_subscr.23', 'data', 'i']
    d = call dist($310binary_subscr.20, $324binary_subscr.23, func=dist, args=[Var($310binary_subscr.20, pynndescent_.py:148), Var($324binary_subscr.23, pynndescent_.py:148)], kws=(), vararg=None, varkwarg=None, target=None) ['$310binary_subscr.20', '$324binary_subscr.23', 'd', 'dist']
    $350load_global.25 = global(checked_flagged_heap_push: CPUDispatcher(<function checked_flagged_heap_push at 0x7fe136b2b420>)) ['$350load_global.25']
    $const364.28.3 = const(int, 1)           ['$const364.28.3']
    $366binary_subscr.29 = getitem(value=heap, index=$const364.28.3, fn=<built-in function getitem>) ['$366binary_subscr.29', '$const364.28.3', 'heap']
    $378binary_subscr.31 = getitem(value=$366binary_subscr.29, index=i, fn=<built-in function getitem>) ['$366binary_subscr.29', '$378binary_subscr.31', 'i']
    $const390.33.1 = const(int, 0)           ['$const390.33.1']
    $392binary_subscr.34 = getitem(value=heap, index=$const390.33.1, fn=<built-in function getitem>) ['$392binary_subscr.34', '$const390.33.1', 'heap']
    $404binary_subscr.36 = getitem(value=$392binary_subscr.34, index=i, fn=<built-in function getitem>) ['$392binary_subscr.34', '$404binary_subscr.36', 'i']
    $const416.38.4 = const(int, 2)           ['$const416.38.4']
    $418binary_subscr.39 = getitem(value=heap, index=$const416.38.4, fn=<built-in function getitem>) ['$418binary_subscr.39', '$const416.38.4', 'heap']
    $430binary_subscr.41 = getitem(value=$418binary_subscr.39, index=i, fn=<built-in function getitem>) ['$418binary_subscr.39', '$430binary_subscr.41', 'i']
    $444load_global.44 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$444load_global.44']
    $456load_attr.46 = getattr(value=$444load_global.44, attr=uint8) ['$444load_global.44', '$456load_attr.46']
    $const466.47.3 = const(int, 1)           ['$const466.47.3']
    $472call.48 = call $456load_attr.46($const466.47.3, func=$456load_attr.46, args=[Var($const466.47.3, pynndescent_.py:150)], kws=(), vararg=None, varkwarg=None, target=None) ['$456load_attr.46', '$472call.48', '$const466.47.3']
    $486call.49 = call $350load_global.25($378binary_subscr.31, $404binary_subscr.36, $430binary_subscr.41, d, idx, $472call.48, func=$350load_global.25, args=[Var($378binary_subscr.31, pynndescent_.py:150), Var($404binary_subscr.36, pynndescent_.py:150), Var($430binary_subscr.41, pynndescent_.py:150), Var(d, pynndescent_.py:148), Var(idx, pynndescent_.py:147), Var($472call.48, pynndescent_.py:150)], kws=(), vararg=None, varkwarg=None, target=None) ['$350load_global.25', '$378binary_subscr.31', '$404binary_subscr.36', '$430binary_subscr.41', '$472call.48', '$486call.49', 'd', 'idx']
    jump 204                                 []
label 500:
    jump 56                                  []
label 502:
    $const502.0.0 = const(NoneType, None)    ['$const502.0.0']
    $504return_value.1 = cast(value=$const502.0.0) ['$504return_value.1', '$const502.0.0']
    return $504return_value.1                ['$504return_value.1']
label 0:
    _152binary_subscr_11_1 = arg(0, name=_152binary_subscr_11_1) ['_152binary_subscr_11_1']
    $const6.1.1 = const(float, 0.0)          ['$const6.1.1']
    $8compare_op.2 = _152binary_subscr_11_1 >= $const6.1.1 ['$8compare_op.2', '$const6.1.1', '_152binary_subscr_11_1']
    $14return_value.3 = cast(value=$8compare_op.2) ['$14return_value.3', '$8compare_op.2']
    return $14return_value.3                 ['$14return_value.3']
label 0:
    arr = arg(0, name=arr)                   ['arr']
    c = freevar(zero: 0)                     ['c']
    $10load_global.1 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$10load_global.1']
    $22load_attr.3 = getattr(value=$10load_global.1, attr=nditer) ['$10load_global.1', '$22load_attr.3']
    $38call.5 = call $22load_attr.3(arr, func=$22load_attr.3, args=[Var(arr, old_arraymath.py:165)], kws=(), vararg=None, varkwarg=None, target=None) ['$22load_attr.3', '$38call.5', 'arr']
    $48get_iter.6 = getiter(value=$38call.5) ['$38call.5', '$48get_iter.6']
    $phi50.0 = $48get_iter.6                 ['$48get_iter.6', '$phi50.0']
    jump 50                                  []
label 50:
    $50for_iter.1 = iternext(value=$phi50.0) ['$50for_iter.1', '$phi50.0']
    $50for_iter.2 = pair_first(value=$50for_iter.1) ['$50for_iter.1', '$50for_iter.2']
    $50for_iter.3 = pair_second(value=$50for_iter.1) ['$50for_iter.1', '$50for_iter.3']
    $phi52.1 = $50for_iter.2                 ['$50for_iter.2', '$phi52.1']
    branch $50for_iter.3, 52, 102            ['$50for_iter.3']
label 52:
    v = $phi52.1                             ['$phi52.1', 'v']
    $58load_method.5 = getattr(value=v, attr=item) ['$58load_method.5', 'v']
    $84call.6 = call $58load_method.5(func=$58load_method.5, args=[], kws=(), vararg=None, varkwarg=None, target=None) ['$58load_method.5', '$84call.6']
    $binop_iadd94.7 = inplace_binop(fn=<built-in function iadd>, immutable_fn=<built-in function add>, lhs=c, rhs=$84call.6, static_lhs=Undefined, static_rhs=Undefined) ['$84call.6', '$binop_iadd94.7', 'c']
    c = $binop_iadd94.7                      ['$binop_iadd94.7', 'c']
    jump 50                                  []
label 102:
    $104return_value.1 = cast(value=c)       ['$104return_value.1', 'c']
    return $104return_value.1                ['$104return_value.1']
label 0:
    current_graph = arg(0, name=current_graph) ['current_graph']
    data = arg(1, name=data)                 ['data']
    n_neighbors = arg(2, name=n_neighbors)   ['n_neighbors']
    rng_state = arg(3, name=rng_state)       ['rng_state']
    max_candidates = arg(4, name=max_candidates) ['max_candidates']
    dist = arg(5, name=dist)                 ['dist']
    n_iters = arg(6, name=n_iters)           ['n_iters']
    delta = arg(7, name=delta)               ['delta']
    verbose = arg(8, name=verbose)           ['verbose']
    $6load_attr.1 = getattr(value=data, attr=shape) ['$6load_attr.1', 'data']
    $const16.2.1 = const(int, 0)             ['$const16.2.1']
    n_vertices = getitem(value=$6load_attr.1, index=$const16.2.1, fn=<built-in function getitem>) ['$6load_attr.1', '$const16.2.1', 'n_vertices']
    block_size = const(int, 16384)           ['block_size']
    n_blocks = n_vertices // block_size      ['block_size', 'n_blocks', 'n_vertices']
    $44load_global.8 = global(numba: <module 'numba' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/__init__.py'>) ['$44load_global.8']
    $56load_attr.10 = getattr(value=$44load_global.8, attr=get_num_threads) ['$44load_global.8', '$56load_attr.10']
    n_threads = call $56load_attr.10(func=$56load_attr.10, args=[], kws=(), vararg=None, varkwarg=None, target=None) ['$56load_attr.10', 'n_threads']
    $82load_global.12 = global(range: <class 'range'>) ['$82load_global.12']
    $100call.15 = call $82load_global.12(n_iters, func=$82load_global.12, args=[Var(n_iters, pynndescent_.py:234)], kws=(), vararg=None, varkwarg=None, target=None) ['$100call.15', '$82load_global.12', 'n_iters']
    $110get_iter.16 = getiter(value=$100call.15) ['$100call.15', '$110get_iter.16']
    $phi112.0 = $110get_iter.16              ['$110get_iter.16', '$phi112.0']
    jump 112                                 []
label 112:
    $112for_iter.1 = iternext(value=$phi112.0) ['$112for_iter.1', '$phi112.0']
    $112for_iter.2 = pair_first(value=$112for_iter.1) ['$112for_iter.1', '$112for_iter.2']
    $112for_iter.3 = pair_second(value=$112for_iter.1) ['$112for_iter.1', '$112for_iter.3']
    $phi114.1 = $112for_iter.2               ['$112for_iter.2', '$phi114.1']
    branch $112for_iter.3, 114, 346          ['$112for_iter.3']
label 114:
    n = $phi114.1                            ['$phi114.1', 'n']
    bool118 = global(bool: <class 'bool'>)   ['bool118']
    $118pred = call bool118(verbose, func=bool118, args=(Var(verbose, pynndescent_.py:234),), kws=(), vararg=None, varkwarg=None, target=None) ['$118pred', 'bool118', 'verbose']
    branch $118pred, 120, 162                ['$118pred']
label 120:
    $120load_global.1 = global(print: <function main.<locals>.dual_print at 0x7fe13772e840>) ['$120load_global.1']
    $const132.3.3 = const(str, 	)            ['$const132.3.3']
    $const136.5.4 = const(int, 1)            ['$const136.5.4']
    $binop_add138.6 = n + $const136.5.4      ['$binop_add138.6', '$const136.5.4', 'n']
    $const142.7.5 = const(str,  / )          ['$const142.7.5']
    $150call.9 = call $120load_global.1($const132.3.3, $binop_add138.6, $const142.7.5, n_iters, func=$120load_global.1, args=[Var($const132.3.3, pynndescent_.py:253), Var($binop_add138.6, pynndescent_.py:253), Var($const142.7.5, pynndescent_.py:253), Var(n_iters, pynndescent_.py:234)], kws=(), vararg=None, varkwarg=None, target=None) ['$120load_global.1', '$150call.9', '$binop_add138.6', '$const132.3.3', '$const142.7.5', 'n_iters']
    jump 162                                 []
label 162:
    $162load_global.1 = global(new_build_candidates: CPUDispatcher(<function new_build_candidates at 0x7fe136b2bba0>)) ['$162load_global.1']
    $186call.7 = call $162load_global.1(current_graph, max_candidates, rng_state, n_threads, func=$162load_global.1, args=[Var(current_graph, pynndescent_.py:234), Var(max_candidates, pynndescent_.py:234), Var(rng_state, pynndescent_.py:234), Var(n_threads, pynndescent_.py:249)], kws=(), vararg=None, varkwarg=None, target=None) ['$162load_global.1', '$186call.7', 'current_graph', 'max_candidates', 'n_threads', 'rng_state']
    $196unpack_sequence.10 = exhaust_iter(value=$186call.7, count=2) ['$186call.7', '$196unpack_sequence.10']
    $196unpack_sequence.8 = static_getitem(value=$196unpack_sequence.10, index=0, index_var=None, fn=<built-in function getitem>) ['$196unpack_sequence.10', '$196unpack_sequence.8']
    $196unpack_sequence.9 = static_getitem(value=$196unpack_sequence.10, index=1, index_var=None, fn=<built-in function getitem>) ['$196unpack_sequence.10', '$196unpack_sequence.9']
    new_candidate_neighbors = $196unpack_sequence.8 ['$196unpack_sequence.8', 'new_candidate_neighbors']
    old_candidate_neighbors = $196unpack_sequence.9 ['$196unpack_sequence.9', 'old_candidate_neighbors']
    $204load_global.11 = global(process_candidates: CPUDispatcher(<function process_candidates at 0x7fe136407880>)) ['$204load_global.11']
    c = call $204load_global.11(data, dist, current_graph, new_candidate_neighbors, old_candidate_neighbors, n_blocks, block_size, n_threads, func=$204load_global.11, args=[Var(data, pynndescent_.py:234), Var(dist, pynndescent_.py:234), Var(current_graph, pynndescent_.py:234), Var(new_candidate_neighbors, pynndescent_.py:255), Var(old_candidate_neighbors, pynndescent_.py:255), Var(n_blocks, pynndescent_.py:248), Var(block_size, pynndescent_.py:247), Var(n_threads, pynndescent_.py:249)], kws=(), vararg=None, varkwarg=None, target=None) ['$204load_global.11', 'block_size', 'c', 'current_graph', 'data', 'dist', 'n_blocks', 'n_threads', 'new_candidate_neighbors', 'old_candidate_neighbors']
    $binop_mul254.25 = delta * n_neighbors   ['$binop_mul254.25', 'delta', 'n_neighbors']
    $260load_attr.27 = getattr(value=data, attr=shape) ['$260load_attr.27', 'data']
    $const270.28.1 = const(int, 0)           ['$const270.28.1']
    $272binary_subscr.29 = getitem(value=$260load_attr.27, index=$const270.28.1, fn=<built-in function getitem>) ['$260load_attr.27', '$272binary_subscr.29', '$const270.28.1']
    $binop_mul282.30 = $binop_mul254.25 * $272binary_subscr.29 ['$272binary_subscr.29', '$binop_mul254.25', '$binop_mul282.30']
    $286compare_op.31 = c <= $binop_mul282.30 ['$286compare_op.31', '$binop_mul282.30', 'c']
    bool292 = global(bool: <class 'bool'>)   ['bool292']
    $292pred = call bool292($286compare_op.31, func=bool292, args=(Var($286compare_op.31, pynndescent_.py:270),), kws=(), vararg=None, varkwarg=None, target=None) ['$286compare_op.31', '$292pred', 'bool292']
    branch $292pred, 294, 344                ['$292pred']
label 294:
    bool296 = global(bool: <class 'bool'>)   ['bool296']
    $296pred = call bool296(verbose, func=bool296, args=(Var(verbose, pynndescent_.py:234),), kws=(), vararg=None, varkwarg=None, target=None) ['$296pred', 'bool296', 'verbose']
    branch $296pred, 298, 338                ['$296pred']
label 298:
    $298load_global.1 = global(print: <function main.<locals>.dual_print at 0x7fe13772e840>) ['$298load_global.1']
    $const310.3.6 = const(str, 	Stopping threshold met -- exiting after) ['$const310.3.6']
    $const314.5.4 = const(int, 1)            ['$const314.5.4']
    $binop_add316.6 = n + $const314.5.4      ['$binop_add316.6', '$const314.5.4', 'n']
    $const320.7.7 = const(str, iterations)   ['$const320.7.7']
    $326call.8 = call $298load_global.1($const310.3.6, $binop_add316.6, $const320.7.7, func=$298load_global.1, args=[Var($const310.3.6, pynndescent_.py:272), Var($binop_add316.6, pynndescent_.py:272), Var($const320.7.7, pynndescent_.py:272)], kws=(), vararg=None, varkwarg=None, target=None) ['$298load_global.1', '$326call.8', '$binop_add316.6', '$const310.3.6', '$const320.7.7']
    jump 338                                 []
label 338:
    $const340.1.0 = const(NoneType, None)    ['$const340.1.0']
    $342return_value.2 = cast(value=$const340.1.0) ['$342return_value.2', '$const340.1.0']
    return $342return_value.2                ['$342return_value.2']
label 344:
    jump 112                                 []
label 346:
    $const346.0.0 = const(NoneType, None)    ['$const346.0.0']
    $348return_value.1 = cast(value=$const346.0.0) ['$348return_value.1', '$const346.0.0']
    return $348return_value.1                ['$348return_value.1']
label 0:
    current_graph = arg(0, name=current_graph) ['current_graph']
    data = arg(1, name=data)                 ['data']
    n_neighbors = arg(2, name=n_neighbors)   ['n_neighbors']
    rng_state = arg(3, name=rng_state)       ['rng_state']
    max_candidates = arg(4, name=max_candidates) ['max_candidates']
    dist = arg(5, name=dist)                 ['dist']
    n_iters = arg(6, name=n_iters)           ['n_iters']
    delta = arg(7, name=delta)               ['delta']
    verbose = arg(8, name=verbose)           ['verbose']
    $8load_attr.1 = getattr(value=data, attr=shape) ['$8load_attr.1', 'data']
    $const18.2.1 = const(int, 0)             ['$const18.2.1']
    n_vertices = getitem(value=$8load_attr.1, index=$const18.2.1, fn=<built-in function getitem>) ['$8load_attr.1', '$const18.2.1', 'n_vertices']
    block_size = const(int, 16384)           ['block_size']
    n_blocks = n_vertices // block_size      ['block_size', 'n_blocks', 'n_vertices']
    $46load_global.8 = global(numba: <module 'numba' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/__init__.py'>) ['$46load_global.8']
    $58load_attr.10 = getattr(value=$46load_global.8, attr=get_num_threads) ['$46load_global.8', '$58load_attr.10']
    n_threads = call $58load_attr.10(func=$58load_attr.10, args=[], kws=(), vararg=None, varkwarg=None, target=None) ['$58load_attr.10', 'n_threads']
    $86build_tuple.13 = build_tuple(items=[Var(current_graph, pynndescent_.py:276)]) ['$86build_tuple.13', 'current_graph']
    $const88.14.3 = const(code, <code object <listcomp> at 0x7fe136dd9cf0, file "/root/subliminal-learning/.venv/lib/python3.11/site-packages/pynndescent/pynndescent_.py", line 293>) ['$const88.14.3']
    $90make_function.15 = make_function(name=None, code=<code object <listcomp> at 0x7fe136dd9cf0, file "/root/subliminal-learning/.venv/lib/python3.11/site-packages/pynndescent/pynndescent_.py", line 293>, closure=$86build_tuple.13, defaults=None) ['$86build_tuple.13', '$90make_function.15']
    $92load_global.16 = global(range: <class 'range'>) ['$92load_global.16']
    $const106.19.1 = const(int, 0)           ['$const106.19.1']
    $108binary_subscr.20 = getitem(value=current_graph, index=$const106.19.1, fn=<built-in function getitem>) ['$108binary_subscr.20', '$const106.19.1', 'current_graph']
    $118load_attr.21 = getattr(value=$108binary_subscr.20, attr=shape) ['$108binary_subscr.20', '$118load_attr.21']
    $const128.22.1 = const(int, 0)           ['$const128.22.1']
    $130binary_subscr.23 = getitem(value=$118load_attr.21, index=$const128.22.1, fn=<built-in function getitem>) ['$118load_attr.21', '$130binary_subscr.23', '$const128.22.1']
    $144call.24 = call $92load_global.16($130binary_subscr.23, func=$92load_global.16, args=[Var($130binary_subscr.23, pynndescent_.py:295)], kws=(), vararg=None, varkwarg=None, target=None) ['$130binary_subscr.23', '$144call.24', '$92load_global.16']
    $154get_iter.25 = getiter(value=$144call.24) ['$144call.24', '$154get_iter.25']
    in_graph = call $90make_function.15($154get_iter.25, func=$90make_function.15, args=[Var($154get_iter.25, pynndescent_.py:293)], kws=(), vararg=None, varkwarg=None, target=None) ['$154get_iter.25', '$90make_function.15', 'in_graph']
    $172load_global.27 = global(range: <class 'range'>) ['$172load_global.27']
    $190call.30 = call $172load_global.27(n_iters, func=$172load_global.27, args=[Var(n_iters, pynndescent_.py:276)], kws=(), vararg=None, varkwarg=None, target=None) ['$172load_global.27', '$190call.30', 'n_iters']
    $200get_iter.31 = getiter(value=$190call.30) ['$190call.30', '$200get_iter.31']
    $phi202.0 = $200get_iter.31              ['$200get_iter.31', '$phi202.0']
    jump 202                                 []
label 202:
    $202for_iter.1 = iternext(value=$phi202.0) ['$202for_iter.1', '$phi202.0']
    $202for_iter.2 = pair_first(value=$202for_iter.1) ['$202for_iter.1', '$202for_iter.2']
    $202for_iter.3 = pair_second(value=$202for_iter.1) ['$202for_iter.1', '$202for_iter.3']
    $phi204.1 = $202for_iter.2               ['$202for_iter.2', '$phi204.1']
    branch $202for_iter.3, 204, 646          ['$202for_iter.3']
label 204:
    n = $phi204.1                            ['$phi204.1', 'n']
    bool208 = global(bool: <class 'bool'>)   ['bool208']
    $208pred = call bool208(verbose, func=bool208, args=(Var(verbose, pynndescent_.py:276),), kws=(), vararg=None, varkwarg=None, target=None) ['$208pred', 'bool208', 'verbose']
    branch $208pred, 210, 252                ['$208pred']
label 210:
    $210load_global.1 = global(print: <function main.<locals>.dual_print at 0x7fe13772e840>) ['$210load_global.1']
    $const222.3.4 = const(str, 	)            ['$const222.3.4']
    $const226.5.5 = const(int, 1)            ['$const226.5.5']
    $binop_add228.6 = n + $const226.5.5      ['$binop_add228.6', '$const226.5.5', 'n']
    $const232.7.6 = const(str,  / )          ['$const232.7.6']
    $240call.9 = call $210load_global.1($const222.3.4, $binop_add228.6, $const232.7.6, n_iters, func=$210load_global.1, args=[Var($const222.3.4, pynndescent_.py:300), Var($binop_add228.6, pynndescent_.py:300), Var($const232.7.6, pynndescent_.py:300), Var(n_iters, pynndescent_.py:276)], kws=(), vararg=None, varkwarg=None, target=None) ['$210load_global.1', '$240call.9', '$binop_add228.6', '$const222.3.4', '$const232.7.6', 'n_iters']
    jump 252                                 []
label 252:
    $252load_global.1 = global(new_build_candidates: CPUDispatcher(<function new_build_candidates at 0x7fe136b2bba0>)) ['$252load_global.1']
    $276call.7 = call $252load_global.1(current_graph, max_candidates, rng_state, n_threads, func=$252load_global.1, args=[Var(current_graph, pynndescent_.py:276), Var(max_candidates, pynndescent_.py:276), Var(rng_state, pynndescent_.py:276), Var(n_threads, pynndescent_.py:291)], kws=(), vararg=None, varkwarg=None, target=None) ['$252load_global.1', '$276call.7', 'current_graph', 'max_candidates', 'n_threads', 'rng_state']
    $286unpack_sequence.10 = exhaust_iter(value=$276call.7, count=2) ['$276call.7', '$286unpack_sequence.10']
    $286unpack_sequence.8 = static_getitem(value=$286unpack_sequence.10, index=0, index_var=None, fn=<built-in function getitem>) ['$286unpack_sequence.10', '$286unpack_sequence.8']
    $286unpack_sequence.9 = static_getitem(value=$286unpack_sequence.10, index=1, index_var=None, fn=<built-in function getitem>) ['$286unpack_sequence.10', '$286unpack_sequence.9']
    new_candidate_neighbors = $286unpack_sequence.8 ['$286unpack_sequence.8', 'new_candidate_neighbors']
    old_candidate_neighbors = $286unpack_sequence.9 ['$286unpack_sequence.9', 'old_candidate_neighbors']
    c = const(int, 0)                        ['c']
    $298load_global.12 = global(range: <class 'range'>) ['$298load_global.12']
    $const312.15.5 = const(int, 1)           ['$const312.15.5']
    $binop_add314.16 = n_blocks + $const312.15.5 ['$binop_add314.16', '$const312.15.5', 'n_blocks']
    $322call.17 = call $298load_global.12($binop_add314.16, func=$298load_global.12, args=[Var($binop_add314.16, pynndescent_.py:307)], kws=(), vararg=None, varkwarg=None, target=None) ['$298load_global.12', '$322call.17', '$binop_add314.16']
    $332get_iter.18 = getiter(value=$322call.17) ['$322call.17', '$332get_iter.18']
    $phi334.1 = $332get_iter.18              ['$332get_iter.18', '$phi334.1']
    jump 334                                 []
label 334:
    $334for_iter.2 = iternext(value=$phi334.1) ['$334for_iter.2', '$phi334.1']
    $334for_iter.3 = pair_first(value=$334for_iter.2) ['$334for_iter.2', '$334for_iter.3']
    $334for_iter.4 = pair_second(value=$334for_iter.2) ['$334for_iter.2', '$334for_iter.4']
    $phi336.2 = $334for_iter.3               ['$334for_iter.3', '$phi336.2']
    branch $334for_iter.4, 336, 548          ['$334for_iter.4']
label 336:
    i = $phi336.2                            ['$phi336.2', 'i']
    block_start = i * block_size             ['block_size', 'block_start', 'i']
    $348load_global.6 = global(min: <built-in function min>) ['$348load_global.6']
    $const364.10.5 = const(int, 1)           ['$const364.10.5']
    $binop_add366.11 = i + $const364.10.5    ['$binop_add366.11', '$const364.10.5', 'i']
    $binop_mul372.13 = $binop_add366.11 * block_size ['$binop_add366.11', '$binop_mul372.13', 'block_size']
    block_end = call $348load_global.6(n_vertices, $binop_mul372.13, func=$348load_global.6, args=[Var(n_vertices, pynndescent_.py:288), Var($binop_mul372.13, pynndescent_.py:309)], kws=(), vararg=None, varkwarg=None, target=None) ['$348load_global.6', '$binop_mul372.13', 'block_end', 'n_vertices']
    $398build_slice.18 = global(slice: <class 'slice'>) ['$398build_slice.18']
    $398build_slice.19 = call $398build_slice.18(block_start, block_end, func=$398build_slice.18, args=(Var(block_start, pynndescent_.py:308), Var(block_end, pynndescent_.py:309)), kws=(), vararg=None, varkwarg=None, target=None) ['$398build_slice.18', '$398build_slice.19', 'block_end', 'block_start']
    new_candidate_block = getitem(value=new_candidate_neighbors, index=$398build_slice.19, fn=<built-in function getitem>) ['$398build_slice.19', 'new_candidate_block', 'new_candidate_neighbors']
    $418build_slice.24 = global(slice: <class 'slice'>) ['$418build_slice.24']
    $418build_slice.25 = call $418build_slice.24(block_start, block_end, func=$418build_slice.24, args=(Var(block_start, pynndescent_.py:308), Var(block_end, pynndescent_.py:309)), kws=(), vararg=None, varkwarg=None, target=None) ['$418build_slice.24', '$418build_slice.25', 'block_end', 'block_start']
    old_candidate_block = getitem(value=old_candidate_neighbors, index=$418build_slice.25, fn=<built-in function getitem>) ['$418build_slice.25', 'old_candidate_block', 'old_candidate_neighbors']
    $const434.28.5 = const(int, 1)           ['$const434.28.5']
    $436binary_subscr.29 = getitem(value=current_graph, index=$const434.28.5, fn=<built-in function getitem>) ['$436binary_subscr.29', '$const434.28.5', 'current_graph']
    $const446.30.0 = const(NoneType, None)   ['$const446.30.0']
    $const448.31.0 = const(NoneType, None)   ['$const448.31.0']
    $450build_slice.32 = global(slice: <class 'slice'>) ['$450build_slice.32']
    $450build_slice.33 = call $450build_slice.32($const446.30.0, $const448.31.0, func=$450build_slice.32, args=(Var($const446.30.0, pynndescent_.py:313), Var($const448.31.0, pynndescent_.py:313)), kws=(), vararg=None, varkwarg=None, target=None) ['$450build_slice.32', '$450build_slice.33', '$const446.30.0', '$const448.31.0']
    $const452.34.1 = const(int, 0)           ['$const452.34.1']
    $454build_tuple.35 = build_tuple(items=[Var($450build_slice.33, pynndescent_.py:313), Var($const452.34.1, pynndescent_.py:313)]) ['$450build_slice.33', '$454build_tuple.35', '$const452.34.1']
    dist_thresholds = getitem(value=$436binary_subscr.29, index=$454build_tuple.35, fn=<built-in function getitem>) ['$436binary_subscr.29', '$454build_tuple.35', 'dist_thresholds']
    $468load_global.37 = global(generate_graph_updates: CPUDispatcher(<function generate_graph_updates at 0x7fe136775260>)) ['$468load_global.37']
    updates = call $468load_global.37(new_candidate_block, old_candidate_block, dist_thresholds, data, dist, func=$468load_global.37, args=[Var(new_candidate_block, pynndescent_.py:311), Var(old_candidate_block, pynndescent_.py:312), Var(dist_thresholds, pynndescent_.py:313), Var(data, pynndescent_.py:276), Var(dist, pynndescent_.py:276)], kws=(), vararg=None, varkwarg=None, target=None) ['$468load_global.37', 'data', 'dist', 'dist_thresholds', 'new_candidate_block', 'old_candidate_block', 'updates']
    $508load_global.46 = global(apply_graph_updates_high_memory: CPUDispatcher(<function apply_graph_updates_high_memory at 0x7fe136cb3380>)) ['$508load_global.46']
    $530call.51 = call $508load_global.46(current_graph, updates, in_graph, func=$508load_global.46, args=[Var(current_graph, pynndescent_.py:276), Var(updates, pynndescent_.py:315), Var(in_graph, pynndescent_.py:293)], kws=(), vararg=None, varkwarg=None, target=None) ['$508load_global.46', '$530call.51', 'current_graph', 'in_graph', 'updates']
    $binop_iadd540.52 = inplace_binop(fn=<built-in function iadd>, immutable_fn=<built-in function add>, lhs=c, rhs=$530call.51, static_lhs=Undefined, static_rhs=Undefined) ['$530call.51', '$binop_iadd540.52', 'c']
    c = $binop_iadd540.52                    ['$binop_iadd540.52', 'c']
    jump 334                                 []
label 548:
    $binop_mul554.4 = delta * n_neighbors    ['$binop_mul554.4', 'delta', 'n_neighbors']
    $560load_attr.6 = getattr(value=data, attr=shape) ['$560load_attr.6', 'data']
    $const570.7.1 = const(int, 0)            ['$const570.7.1']
    $572binary_subscr.8 = getitem(value=$560load_attr.6, index=$const570.7.1, fn=<built-in function getitem>) ['$560load_attr.6', '$572binary_subscr.8', '$const570.7.1']
    $binop_mul582.9 = $binop_mul554.4 * $572binary_subscr.8 ['$572binary_subscr.8', '$binop_mul554.4', '$binop_mul582.9']
    $586compare_op.10 = c <= $binop_mul582.9 ['$586compare_op.10', '$binop_mul582.9', 'c']
    bool592 = global(bool: <class 'bool'>)   ['bool592']
    $592pred = call bool592($586compare_op.10, func=bool592, args=(Var($586compare_op.10, pynndescent_.py:321),), kws=(), vararg=None, varkwarg=None, target=None) ['$586compare_op.10', '$592pred', 'bool592']
    branch $592pred, 594, 644                ['$592pred']
label 594:
    bool596 = global(bool: <class 'bool'>)   ['bool596']
    $596pred = call bool596(verbose, func=bool596, args=(Var(verbose, pynndescent_.py:276),), kws=(), vararg=None, varkwarg=None, target=None) ['$596pred', 'bool596', 'verbose']
    branch $596pred, 598, 638                ['$596pred']
label 598:
    $598load_global.1 = global(print: <function main.<locals>.dual_print at 0x7fe13772e840>) ['$598load_global.1']
    $const610.3.7 = const(str, 	Stopping threshold met -- exiting after) ['$const610.3.7']
    $const614.5.5 = const(int, 1)            ['$const614.5.5']
    $binop_add616.6 = n + $const614.5.5      ['$binop_add616.6', '$const614.5.5', 'n']
    $const620.7.8 = const(str, iterations)   ['$const620.7.8']
    $626call.8 = call $598load_global.1($const610.3.7, $binop_add616.6, $const620.7.8, func=$598load_global.1, args=[Var($const610.3.7, pynndescent_.py:323), Var($binop_add616.6, pynndescent_.py:323), Var($const620.7.8, pynndescent_.py:323)], kws=(), vararg=None, varkwarg=None, target=None) ['$598load_global.1', '$626call.8', '$binop_add616.6', '$const610.3.7', '$const620.7.8']
    jump 638                                 []
label 638:
    $const640.1.0 = const(NoneType, None)    ['$const640.1.0']
    $642return_value.2 = cast(value=$const640.1.0) ['$642return_value.2', '$const640.1.0']
    return $642return_value.2                ['$642return_value.2']
label 644:
    jump 202                                 []
label 646:
    $const646.0.0 = const(NoneType, None)    ['$const646.0.0']
    $648return_value.1 = cast(value=$const646.0.0) ['$648return_value.1', '$const646.0.0']
    return $648return_value.1                ['$648return_value.1']
label 0:
    implicit0 = arg(0, name=implicit0)       ['implicit0']
    $6build_list.0 = build_list(items=[])    ['$6build_list.0']
    $phi12.0 = $6build_list.0                ['$6build_list.0', '$phi12.0']
    $phi128.0 = $6build_list.0               ['$6build_list.0', '$phi128.0']
    $phi10.1 = implicit0                     ['$phi10.1', 'implicit0']
    jump 10                                  []
label 10:
    $10for_iter.2 = iternext(value=$phi10.1) ['$10for_iter.2', '$phi10.1']
    $10for_iter.3 = pair_first(value=$10for_iter.2) ['$10for_iter.2', '$10for_iter.3']
    $10for_iter.4 = pair_second(value=$10for_iter.2) ['$10for_iter.2', '$10for_iter.4']
    $phi12.2 = $10for_iter.3                 ['$10for_iter.3', '$phi12.2']
    branch $10for_iter.4, 12, 128            ['$10for_iter.4']
label 12:
    i = $phi12.2                             ['$phi12.2', 'i']
    $14load_global.3 = global(set: <class 'set'>) ['$14load_global.3']
    $26load_deref.5 = freevar(current_graph: None) ['$26load_deref.5']
    $const28.6.0 = const(int, 0)             ['$const28.6.0']
    $30binary_subscr.7 = getitem(value=$26load_deref.5, index=$const28.6.0, fn=<built-in function getitem>) ['$26load_deref.5', '$30binary_subscr.7', '$const28.6.0']
    $42binary_subscr.9 = getitem(value=$30binary_subscr.7, index=i, fn=<built-in function getitem>) ['$30binary_subscr.7', '$42binary_subscr.9', 'i']
    $52load_method.11 = getattr(value=$42binary_subscr.9, attr=astype) ['$42binary_subscr.9', '$52load_method.11']
    $74load_global.12 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$74load_global.12']
    $86load_attr.13 = getattr(value=$74load_global.12, attr=int64) ['$74load_global.12', '$86load_attr.13']
    $100call.14 = call $52load_method.11($86load_attr.13, func=$52load_method.11, args=[Var($86load_attr.13, pynndescent_.py:294)], kws=(), vararg=None, varkwarg=None, target=None) ['$100call.14', '$52load_method.11', '$86load_attr.13']
    $114call.15 = call $14load_global.3($100call.14, func=$14load_global.3, args=[Var($100call.14, pynndescent_.py:294)], kws=(), vararg=None, varkwarg=None, target=None) ['$100call.14', '$114call.15', '$14load_global.3']
    $124list_append.16 = getattr(value=$phi12.0, attr=append) ['$124list_append.16', '$phi12.0']
    $124list_append.17 = call $124list_append.16($114call.15, func=$124list_append.16, args=(Var($114call.15, pynndescent_.py:294),), kws=(), vararg=None, varkwarg=None, target=None) ['$114call.15', '$124list_append.16', '$124list_append.17']
    jump 10                                  []
label 128:
    $128return_value.1 = cast(value=$phi128.0) ['$128return_value.1', '$phi128.0']
    return $128return_value.1                ['$128return_value.1']
label 0:
    indices = arg(0, name=indices)           ['indices']
    distances = arg(1, name=distances)       ['distances']
    $4load_global.0 = global(numba: <module 'numba' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numba/__init__.py'>) ['$4load_global.0']
    $16load_attr.2 = getattr(value=$4load_global.0, attr=prange) ['$16load_attr.2', '$4load_global.0']
    $28load_attr.4 = getattr(value=indices, attr=shape) ['$28load_attr.4', 'indices']
    $const38.5.1 = const(int, 0)             ['$const38.5.1']
    $40binary_subscr.6 = getitem(value=$28load_attr.4, index=$const38.5.1, fn=<built-in function getitem>) ['$28load_attr.4', '$40binary_subscr.6', '$const38.5.1']
    $54call.7 = call $16load_attr.2($40binary_subscr.6, func=$16load_attr.2, args=[Var($40binary_subscr.6, utils.py:247)], kws=(), vararg=None, varkwarg=None, target=None) ['$16load_attr.2', '$40binary_subscr.6', '$54call.7']
    $64get_iter.8 = getiter(value=$54call.7) ['$54call.7', '$64get_iter.8']
    $phi66.0 = $64get_iter.8                 ['$64get_iter.8', '$phi66.0']
    jump 66                                  []
label 66:
    $66for_iter.1 = iternext(value=$phi66.0) ['$66for_iter.1', '$phi66.0']
    $66for_iter.2 = pair_first(value=$66for_iter.1) ['$66for_iter.1', '$66for_iter.2']
    $66for_iter.3 = pair_second(value=$66for_iter.1) ['$66for_iter.1', '$66for_iter.3']
    $phi68.1 = $66for_iter.2                 ['$66for_iter.2', '$phi68.1']
    branch $66for_iter.3, 68, 338            ['$66for_iter.3']
label 68:
    i = $phi68.1                             ['$phi68.1', 'i']
    $70load_global.2 = global(range: <class 'range'>) ['$70load_global.2']
    $84load_attr.5 = getattr(value=indices, attr=shape) ['$84load_attr.5', 'indices']
    $const94.6.2 = const(int, 1)             ['$const94.6.2']
    $96binary_subscr.7 = getitem(value=$84load_attr.5, index=$const94.6.2, fn=<built-in function getitem>) ['$84load_attr.5', '$96binary_subscr.7', '$const94.6.2']
    $const106.8.2 = const(int, 1)            ['$const106.8.2']
    $binop_sub108.9 = $96binary_subscr.7 - $const106.8.2 ['$96binary_subscr.7', '$binop_sub108.9', '$const106.8.2']
    $const112.10.1 = const(int, 0)           ['$const112.10.1']
    $const114.11.3 = const(int, -1)          ['$const114.11.3']
    $120call.12 = call $70load_global.2($binop_sub108.9, $const112.10.1, $const114.11.3, func=$70load_global.2, args=[Var($binop_sub108.9, utils.py:249), Var($const112.10.1, utils.py:249), Var($const114.11.3, utils.py:249)], kws=(), vararg=None, varkwarg=None, target=None) ['$120call.12', '$70load_global.2', '$binop_sub108.9', '$const112.10.1', '$const114.11.3']
    $130get_iter.13 = getiter(value=$120call.12) ['$120call.12', '$130get_iter.13']
    $phi132.1 = $130get_iter.13              ['$130get_iter.13', '$phi132.1']
    jump 132                                 []
label 132:
    $132for_iter.2 = iternext(value=$phi132.1) ['$132for_iter.2', '$phi132.1']
    $132for_iter.3 = pair_first(value=$132for_iter.2) ['$132for_iter.2', '$132for_iter.3']
    $132for_iter.4 = pair_second(value=$132for_iter.2) ['$132for_iter.2', '$132for_iter.4']
    $phi134.2 = $132for_iter.3               ['$132for_iter.3', '$phi134.2']
    branch $132for_iter.4, 134, 336          ['$132for_iter.4']
label 134:
    j = $phi134.2                            ['$phi134.2', 'j']
    $142build_tuple.6 = build_tuple(items=[Var(i, utils.py:247), Var(j, utils.py:249)]) ['$142build_tuple.6', 'i', 'j']
    $144binary_subscr.7 = getitem(value=indices, index=$142build_tuple.6, fn=<built-in function getitem>) ['$142build_tuple.6', '$144binary_subscr.7', 'indices']
    $const158.10.1 = const(int, 0)           ['$const158.10.1']
    $160build_tuple.11 = build_tuple(items=[Var(i, utils.py:247), Var($const158.10.1, utils.py:250)]) ['$160build_tuple.11', '$const158.10.1', 'i']
    $162binary_subscr.12 = getitem(value=indices, index=$160build_tuple.11, fn=<built-in function getitem>) ['$160build_tuple.11', '$162binary_subscr.12', 'indices']
    $const178.15.1 = const(int, 0)           ['$const178.15.1']
    $180build_tuple.16 = build_tuple(items=[Var(i, utils.py:247), Var($const178.15.1, utils.py:250)]) ['$180build_tuple.16', '$const178.15.1', 'i']
    indices[$180build_tuple.16] = $144binary_subscr.7 ['$144binary_subscr.7', '$180build_tuple.16', 'indices']
    $192build_tuple.20 = build_tuple(items=[Var(i, utils.py:247), Var(j, utils.py:249)]) ['$192build_tuple.20', 'i', 'j']
    indices[$192build_tuple.20] = $162binary_subscr.12 ['$162binary_subscr.12', '$192build_tuple.20', 'indices']
    $204build_tuple.24 = build_tuple(items=[Var(i, utils.py:247), Var(j, utils.py:249)]) ['$204build_tuple.24', 'i', 'j']
    $206binary_subscr.25 = getitem(value=distances, index=$204build_tuple.24, fn=<built-in function getitem>) ['$204build_tuple.24', '$206binary_subscr.25', 'distances']
    $const220.28.1 = const(int, 0)           ['$const220.28.1']
    $222build_tuple.29 = build_tuple(items=[Var(i, utils.py:247), Var($const220.28.1, utils.py:251)]) ['$222build_tuple.29', '$const220.28.1', 'i']
    $224binary_subscr.30 = getitem(value=distances, index=$222build_tuple.29, fn=<built-in function getitem>) ['$222build_tuple.29', '$224binary_subscr.30', 'distances']
    $const240.33.1 = const(int, 0)           ['$const240.33.1']
    $242build_tuple.34 = build_tuple(items=[Var(i, utils.py:247), Var($const240.33.1, utils.py:251)]) ['$242build_tuple.34', '$const240.33.1', 'i']
    distances[$242build_tuple.34] = $206binary_subscr.25 ['$206binary_subscr.25', '$242build_tuple.34', 'distances']
    $254build_tuple.38 = build_tuple(items=[Var(i, utils.py:247), Var(j, utils.py:249)]) ['$254build_tuple.38', 'i', 'j']
    distances[$254build_tuple.38] = $224binary_subscr.30 ['$224binary_subscr.30', '$254build_tuple.38', 'distances']
    $260load_global.39 = global(siftdown: CPUDispatcher(<function siftdown at 0x7fe136b2b880>)) ['$260load_global.39']
    $const276.43.4 = const(NoneType, None)   ['$const276.43.4']
    $280build_slice.45 = global(slice: <class 'slice'>) ['$280build_slice.45']
    $280build_slice.46 = call $280build_slice.45($const276.43.4, j, func=$280build_slice.45, args=(Var($const276.43.4, utils.py:253), Var(j, utils.py:249)), kws=(), vararg=None, varkwarg=None, target=None) ['$280build_slice.45', '$280build_slice.46', '$const276.43.4', 'j']
    $282build_tuple.47 = build_tuple(items=[Var(i, utils.py:247), Var($280build_slice.46, utils.py:253)]) ['$280build_slice.46', '$282build_tuple.47', 'i']
    $284binary_subscr.48 = getitem(value=distances, index=$282build_tuple.47, fn=<built-in function getitem>) ['$282build_tuple.47', '$284binary_subscr.48', 'distances']
    $const298.51.4 = const(NoneType, None)   ['$const298.51.4']
    $302build_slice.53 = global(slice: <class 'slice'>) ['$302build_slice.53']
    $302build_slice.54 = call $302build_slice.53($const298.51.4, j, func=$302build_slice.53, args=(Var($const298.51.4, utils.py:253), Var(j, utils.py:249)), kws=(), vararg=None, varkwarg=None, target=None) ['$302build_slice.53', '$302build_slice.54', '$const298.51.4', 'j']
    $304build_tuple.55 = build_tuple(items=[Var(i, utils.py:247), Var($302build_slice.54, utils.py:253)]) ['$302build_slice.54', '$304build_tuple.55', 'i']
    $306binary_subscr.56 = getitem(value=indices, index=$304build_tuple.55, fn=<built-in function getitem>) ['$304build_tuple.55', '$306binary_subscr.56', 'indices']
    $const316.57.1 = const(int, 0)           ['$const316.57.1']
    $322call.58 = call $260load_global.39($284binary_subscr.48, $306binary_subscr.56, $const316.57.1, func=$260load_global.39, args=[Var($284binary_subscr.48, utils.py:253), Var($306binary_subscr.56, utils.py:253), Var($const316.57.1, utils.py:253)], kws=(), vararg=None, varkwarg=None, target=None) ['$260load_global.39', '$284binary_subscr.48', '$306binary_subscr.56', '$322call.58', '$const316.57.1']
    jump 132                                 []
label 336:
    jump 66                                  []
label 338:
    $342build_tuple.2 = build_tuple(items=[Var(indices, utils.py:226), Var(distances, utils.py:226)]) ['$342build_tuple.2', 'distances', 'indices']
    $344return_value.3 = cast(value=$342build_tuple.2) ['$342build_tuple.2', '$344return_value.3']
    return $344return_value.3                ['$344return_value.3']
label 0:
    heap1 = arg(0, name=heap1)               ['heap1']
    heap2 = arg(1, name=heap2)               ['heap2']
    elt = arg(2, name=elt)                   ['elt']
    $const6.1.1 = const(int, 2)              ['$const6.1.1']
    $binop_mul8.2 = elt * $const6.1.1        ['$binop_mul8.2', '$const6.1.1', 'elt']
    $const12.3.2 = const(int, 1)             ['$const12.3.2']
    $binop_add14.4 = $binop_mul8.2 + $const12.3.2 ['$binop_add14.4', '$binop_mul8.2', '$const12.3.2']
    $20load_attr.6 = getattr(value=heap1, attr=shape) ['$20load_attr.6', 'heap1']
    $const30.7.3 = const(int, 0)             ['$const30.7.3']
    $32binary_subscr.8 = getitem(value=$20load_attr.6, index=$const30.7.3, fn=<built-in function getitem>) ['$20load_attr.6', '$32binary_subscr.8', '$const30.7.3']
    $42compare_op.9 = $binop_add14.4 < $32binary_subscr.8 ['$32binary_subscr.8', '$42compare_op.9', '$binop_add14.4']
    bool48 = global(bool: <class 'bool'>)    ['bool48']
    $48pred = call bool48($42compare_op.9, func=bool48, args=(Var($42compare_op.9, utils.py:207),), kws=(), vararg=None, varkwarg=None, target=None) ['$42compare_op.9', '$48pred', 'bool48']
    branch $48pred, 50, 356                  ['$48pred']
label 50:
    $const52.1.1 = const(int, 2)             ['$const52.1.1']
    $binop_mul54.2 = elt * $const52.1.1      ['$binop_mul54.2', '$const52.1.1', 'elt']
    $const58.3.2 = const(int, 1)             ['$const58.3.2']
    left_child = $binop_mul54.2 + $const58.3.2 ['$binop_mul54.2', '$const58.3.2', 'left_child']
    $const68.6.2 = const(int, 1)             ['$const68.6.2']
    right_child = left_child + $const68.6.2  ['$const68.6.2', 'left_child', 'right_child']
    swap = elt                               ['elt', 'swap']
    $84binary_subscr.11 = getitem(value=heap1, index=swap, fn=<built-in function getitem>) ['$84binary_subscr.11', 'heap1', 'swap']
    $98binary_subscr.14 = getitem(value=heap1, index=left_child, fn=<built-in function getitem>) ['$98binary_subscr.14', 'heap1', 'left_child']
    $108compare_op.15 = $84binary_subscr.11 < $98binary_subscr.14 ['$108compare_op.15', '$84binary_subscr.11', '$98binary_subscr.14']
    bool114 = global(bool: <class 'bool'>)   ['bool114']
    $114pred = call bool114($108compare_op.15, func=bool114, args=(Var($108compare_op.15, utils.py:212),), kws=(), vararg=None, varkwarg=None, target=None) ['$108compare_op.15', '$114pred', 'bool114']
    branch $114pred, 116, 120                ['$114pred']
label 116:
    swap = left_child                        ['left_child', 'swap']
    jump 120                                 []
label 120:
    $124load_attr.2 = getattr(value=heap1, attr=shape) ['$124load_attr.2', 'heap1']
    $const134.3.3 = const(int, 0)            ['$const134.3.3']
    $136binary_subscr.4 = getitem(value=$124load_attr.2, index=$const134.3.3, fn=<built-in function getitem>) ['$124load_attr.2', '$136binary_subscr.4', '$const134.3.3']
    $146compare_op.5 = right_child < $136binary_subscr.4 ['$136binary_subscr.4', '$146compare_op.5', 'right_child']
    bool152 = global(bool: <class 'bool'>)   ['bool152']
    $152pred = call bool152($146compare_op.5, func=bool152, args=(Var($146compare_op.5, utils.py:215),), kws=(), vararg=None, varkwarg=None, target=None) ['$146compare_op.5', '$152pred', 'bool152']
    branch $152pred, 154, 194                ['$152pred']
label 154:
    $158binary_subscr.2 = getitem(value=heap1, index=swap, fn=<built-in function getitem>) ['$158binary_subscr.2', 'heap1', 'swap']
    $172binary_subscr.5 = getitem(value=heap1, index=right_child, fn=<built-in function getitem>) ['$172binary_subscr.5', 'heap1', 'right_child']
    $182compare_op.6 = $158binary_subscr.2 < $172binary_subscr.5 ['$158binary_subscr.2', '$172binary_subscr.5', '$182compare_op.6']
    bool188 = global(bool: <class 'bool'>)   ['bool188']
    $188pred = call bool188($182compare_op.6, func=bool188, args=(Var($182compare_op.6, utils.py:215),), kws=(), vararg=None, varkwarg=None, target=None) ['$182compare_op.6', '$188pred', 'bool188']
    branch $188pred, 190, 194                ['$188pred']
label 190:
    swap = right_child                       ['right_child', 'swap']
    jump 194                                 []
label 194:
    $198compare_op.2 = swap == elt           ['$198compare_op.2', 'elt', 'swap']
    bool204 = global(bool: <class 'bool'>)   ['bool204']
    $204pred = call bool204($198compare_op.2, func=bool204, args=(Var($198compare_op.2, utils.py:218),), kws=(), vararg=None, varkwarg=None, target=None) ['$198compare_op.2', '$204pred', 'bool204']
    branch $204pred, 206, 210                ['$204pred']
label 206:
    $const206.0.4 = const(NoneType, None)    ['$const206.0.4']
    $208return_value.1 = cast(value=$const206.0.4) ['$208return_value.1', '$const206.0.4']
    return $208return_value.1                ['$208return_value.1']
label 210:
    $214binary_subscr.2 = getitem(value=heap1, index=swap, fn=<built-in function getitem>) ['$214binary_subscr.2', 'heap1', 'swap']
    $228binary_subscr.5 = getitem(value=heap1, index=elt, fn=<built-in function getitem>) ['$228binary_subscr.5', 'elt', 'heap1']
    heap1[elt] = $214binary_subscr.2         ['$214binary_subscr.2', 'elt', 'heap1']
    heap1[swap] = $228binary_subscr.5        ['$228binary_subscr.5', 'heap1', 'swap']
    $260binary_subscr.12 = getitem(value=heap2, index=swap, fn=<built-in function getitem>) ['$260binary_subscr.12', 'heap2', 'swap']
    $274binary_subscr.15 = getitem(value=heap2, index=elt, fn=<built-in function getitem>) ['$274binary_subscr.15', 'elt', 'heap2']
    heap2[elt] = $260binary_subscr.12        ['$260binary_subscr.12', 'elt', 'heap2']
    heap2[swap] = $274binary_subscr.15       ['$274binary_subscr.15', 'heap2', 'swap']
    elt = swap                               ['elt', 'swap']
    $const308.22.1 = const(int, 2)           ['$const308.22.1']
    $binop_mul310.23 = elt * $const308.22.1  ['$binop_mul310.23', '$const308.22.1', 'elt']
    $const314.24.2 = const(int, 1)           ['$const314.24.2']
    $binop_add316.25 = $binop_mul310.23 + $const314.24.2 ['$binop_add316.25', '$binop_mul310.23', '$const314.24.2']
    $322load_attr.27 = getattr(value=heap1, attr=shape) ['$322load_attr.27', 'heap1']
    $const332.28.3 = const(int, 0)           ['$const332.28.3']
    $334binary_subscr.29 = getitem(value=$322load_attr.27, index=$const332.28.3, fn=<built-in function getitem>) ['$322load_attr.27', '$334binary_subscr.29', '$const332.28.3']
    $344compare_op.30 = $binop_add316.25 < $334binary_subscr.29 ['$334binary_subscr.29', '$344compare_op.30', '$binop_add316.25']
    bool350 = global(bool: <class 'bool'>)   ['bool350']
    $350pred = call bool350($344compare_op.30, func=bool350, args=(Var($344compare_op.30, utils.py:207),), kws=(), vararg=None, varkwarg=None, target=None) ['$344compare_op.30', '$350pred', 'bool350']
    branch $350pred, 50, 352                 ['$350pred']
label 352:
    $const352.0.4 = const(NoneType, None)    ['$const352.0.4']
    $354return_value.1 = cast(value=$const352.0.4) ['$354return_value.1', '$const352.0.4']
    return $354return_value.1                ['$354return_value.1']
label 356:
    $const356.0.4 = const(NoneType, None)    ['$const356.0.4']
    $358return_value.1 = cast(value=$const356.0.4) ['$358return_value.1', '$const356.0.4']
    return $358return_value.1                ['$358return_value.1']
# File: /root/subliminal-learning/.venv/lib/python3.11/site-packages/pynndescent/utils.py
# --- LINE 202 --- 
# label 0
#   heap1 = arg(0, name=heap1)  :: array(float32, 1d, C)
#   heap2 = arg(1, name=heap2)  :: array(int32, 1d, C)
#   elt = arg(2, name=elt)  :: Literal[int](0)
#   elt.2 = elt  :: int64

@numba.njit(cache=True)

# --- LINE 203 --- 

def siftdown(heap1, heap2, elt):

    # --- LINE 204 --- 

    """Restore the heap property for a heap with an out of place element

    # --- LINE 205 --- 

    at position ``elt``. This works with a heap pair where heap1 carries

    # --- LINE 206 --- 

    the weights and heap2 holds the corresponding elements."""

    # --- LINE 207 --- 
    #   $const6.1.1 = const(int, 2)  :: Literal[int](2)
    #   $binop_mul8.2 = elt * $const6.1.1  :: int64
    #   del elt
    #   del $const6.1.1
    #   $const12.3.2 = const(int, 1)  :: Literal[int](1)
    #   $binop_add14.4 = $binop_mul8.2 + $const12.3.2  :: int64
    #   del $const12.3.2
    #   del $binop_mul8.2
    #   $20load_attr.6 = getattr(value=heap1, attr=shape)  :: UniTuple(int64 x 1)
    #   $const30.7.3 = const(int, 0)  :: Literal[int](0)
    #   $32binary_subscr.8 = static_getitem(value=$20load_attr.6, index=0, index_var=$const30.7.3, fn=<built-in function getitem>)  :: int64
    #   del $const30.7.3
    #   del $20load_attr.6
    #   $42compare_op.9 = $binop_add14.4 < $32binary_subscr.8  :: bool
    #   del $binop_add14.4
    #   del $32binary_subscr.8
    #   bool48 = global(bool: <class 'bool'>)  :: Function(<class 'bool'>)
    #   $48pred = call bool48($42compare_op.9, func=bool48, args=(Var($42compare_op.9, utils.py:207),), kws=(), vararg=None, varkwarg=None, target=None)  :: (bool,) -> bool
    #   del bool48
    #   del $42compare_op.9
    #   branch $48pred, 50, 356
    #   $const308.22.1 = const(int, 2)  :: Literal[int](2)
    #   $binop_mul310.23 = elt.1 * $const308.22.1  :: int64
    #   del elt.1
    #   del $const308.22.1
    #   $const314.24.2 = const(int, 1)  :: Literal[int](1)
    #   $binop_add316.25 = $binop_mul310.23 + $const314.24.2  :: int64
    #   del $const314.24.2
    #   del $binop_mul310.23
    #   $322load_attr.27 = getattr(value=heap1, attr=shape)  :: UniTuple(int64 x 1)
    #   $const332.28.3 = const(int, 0)  :: Literal[int](0)
    #   $334binary_subscr.29 = static_getitem(value=$322load_attr.27, index=0, index_var=$const332.28.3, fn=<built-in function getitem>)  :: int64
    #   del $const332.28.3
    #   del $322load_attr.27
    #   $344compare_op.30 = $binop_add316.25 < $334binary_subscr.29  :: bool
    #   del $binop_add316.25
    #   del $334binary_subscr.29
    #   bool350 = global(bool: <class 'bool'>)  :: Function(<class 'bool'>)
    #   $350pred = call bool350($344compare_op.30, func=bool350, args=(Var($344compare_op.30, utils.py:207),), kws=(), vararg=None, varkwarg=None, target=None)  :: (bool,) -> bool
    #   del bool350
    #   del $344compare_op.30
    #   branch $350pred, 50, 352
    # label 352
    #   del heap2
    #   del heap1
    #   del elt.2
    #   del $350pred
    #   $const352.0.4 = const(NoneType, None)  :: none
    #   $354return_value.1 = cast(value=$const352.0.4)  :: none
    #   del $const352.0.4
    #   return $354return_value.1
    # label 356
    #   del heap2
    #   del heap1
    #   del elt.2
    #   del $48pred
    #   $const356.0.4 = const(NoneType, None)  :: none
    #   $358return_value.1 = cast(value=$const356.0.4)  :: none
    #   del $const356.0.4
    #   return $358return_value.1

    while elt * 2 + 1 < heap1.shape[0]:

        # --- LINE 208 --- 
        # label 50
        #   del $48pred
        #   del $350pred
        #   $const52.1.1 = const(int, 2)  :: Literal[int](2)
        #   $binop_mul54.2 = elt.2 * $const52.1.1  :: int64
        #   del $const52.1.1
        #   $const58.3.2 = const(int, 1)  :: Literal[int](1)
        #   left_child = $binop_mul54.2 + $const58.3.2  :: int64
        #   del $const58.3.2
        #   del $binop_mul54.2

        left_child = elt * 2 + 1

        # --- LINE 209 --- 
        #   $const68.6.2 = const(int, 1)  :: Literal[int](1)
        #   right_child = left_child + $const68.6.2  :: int64
        #   del $const68.6.2

        right_child = left_child + 1

        # --- LINE 210 --- 
        #   swap = elt.2  :: int64
        #   swap.3 = swap  :: int64

        swap = elt

# --- LINE 211 --- 



        # --- LINE 212 --- 
        #   $84binary_subscr.11 = getitem(value=heap1, index=swap, fn=<built-in function getitem>)  :: float32
        #   del swap
        #   $98binary_subscr.14 = getitem(value=heap1, index=left_child, fn=<built-in function getitem>)  :: float32
        #   $108compare_op.15 = $84binary_subscr.11 < $98binary_subscr.14  :: bool
        #   del $98binary_subscr.14
        #   del $84binary_subscr.11
        #   bool114 = global(bool: <class 'bool'>)  :: Function(<class 'bool'>)
        #   $114pred = call bool114($108compare_op.15, func=bool114, args=(Var($108compare_op.15, utils.py:212),), kws=(), vararg=None, varkwarg=None, target=None)  :: (bool,) -> bool
        #   del bool114
        #   del $108compare_op.15
        #   branch $114pred, 116, 120

        if heap1[swap] < heap1[left_child]:

            # --- LINE 213 --- 
            # label 116
            #   del $114pred
            #   swap.1 = left_child  :: int64
            #   del left_child
            #   swap.3 = swap.1  :: int64
            #   del swap.1

            swap = left_child

# --- LINE 214 --- 



        # --- LINE 215 --- 
        #   jump 120
        # label 120
        #   del left_child
        #   del $114pred
        #   swap.4 = swap.3  :: int64
        #   $124load_attr.2 = getattr(value=heap1, attr=shape)  :: UniTuple(int64 x 1)
        #   $const134.3.3 = const(int, 0)  :: Literal[int](0)
        #   $136binary_subscr.4 = static_getitem(value=$124load_attr.2, index=0, index_var=$const134.3.3, fn=<built-in function getitem>)  :: int64
        #   del $const134.3.3
        #   del $124load_attr.2
        #   $146compare_op.5 = right_child < $136binary_subscr.4  :: bool
        #   del $136binary_subscr.4
        #   bool152 = global(bool: <class 'bool'>)  :: Function(<class 'bool'>)
        #   $152pred = call bool152($146compare_op.5, func=bool152, args=(Var($146compare_op.5, utils.py:215),), kws=(), vararg=None, varkwarg=None, target=None)  :: (bool,) -> bool
        #   del bool152
        #   del $146compare_op.5
        #   branch $152pred, 154, 194
        # label 154
        #   del $152pred
        #   swap.4 = swap.3  :: int64
        #   $158binary_subscr.2 = getitem(value=heap1, index=swap.3, fn=<built-in function getitem>)  :: float32
        #   del swap.3
        #   $172binary_subscr.5 = getitem(value=heap1, index=right_child, fn=<built-in function getitem>)  :: float32
        #   $182compare_op.6 = $158binary_subscr.2 < $172binary_subscr.5  :: bool
        #   del $172binary_subscr.5
        #   del $158binary_subscr.2
        #   bool188 = global(bool: <class 'bool'>)  :: Function(<class 'bool'>)
        #   $188pred = call bool188($182compare_op.6, func=bool188, args=(Var($182compare_op.6, utils.py:215),), kws=(), vararg=None, varkwarg=None, target=None)  :: (bool,) -> bool
        #   del bool188
        #   del $182compare_op.6
        #   branch $188pred, 190, 194

        if right_child < heap1.shape[0] and heap1[swap] < heap1[right_child]:

            # --- LINE 216 --- 
            # label 190
            #   del $188pred
            #   swap.2 = right_child  :: int64
            #   del right_child
            #   swap.4 = swap.2  :: int64
            #   del swap.2

            swap = right_child

# --- LINE 217 --- 



        # --- LINE 218 --- 
        #   jump 194
        # label 194
        #   del swap.3
        #   del right_child
        #   del $188pred
        #   del $152pred
        #   $198compare_op.2 = swap.4 == elt.2  :: bool
        #   bool204 = global(bool: <class 'bool'>)  :: Function(<class 'bool'>)
        #   $204pred = call bool204($198compare_op.2, func=bool204, args=(Var($198compare_op.2, utils.py:218),), kws=(), vararg=None, varkwarg=None, target=None)  :: (bool,) -> bool
        #   del bool204
        #   del $198compare_op.2
        #   branch $204pred, 206, 210

        if swap == elt:

            # --- LINE 219 --- 
            # label 206
            #   del swap.4
            #   del heap2
            #   del heap1
            #   del elt.2
            #   del $204pred
            #   $const206.0.4 = const(NoneType, None)  :: none
            #   $208return_value.1 = cast(value=$const206.0.4)  :: none
            #   del $const206.0.4
            #   return $208return_value.1

            break

        # --- LINE 220 --- 

        else:

            # --- LINE 221 --- 
            # label 210
            #   del $204pred
            #   $214binary_subscr.2 = getitem(value=heap1, index=swap.4, fn=<built-in function getitem>)  :: float32
            #   $228binary_subscr.5 = getitem(value=heap1, index=elt.2, fn=<built-in function getitem>)  :: float32
            #   heap1[elt.2] = $214binary_subscr.2  :: (Array(float32, 1, 'C', False, aligned=True), int64, float32) -> none
            #   del $214binary_subscr.2
            #   heap1[swap.4] = $228binary_subscr.5  :: (Array(float32, 1, 'C', False, aligned=True), int64, float32) -> none
            #   del $228binary_subscr.5

            heap1[elt], heap1[swap] = heap1[swap], heap1[elt]

            # --- LINE 222 --- 
            #   $260binary_subscr.12 = getitem(value=heap2, index=swap.4, fn=<built-in function getitem>)  :: int32
            #   $274binary_subscr.15 = getitem(value=heap2, index=elt.2, fn=<built-in function getitem>)  :: int32
            #   heap2[elt.2] = $260binary_subscr.12  :: (Array(int32, 1, 'C', False, aligned=True), int64, int32) -> none
            #   del $260binary_subscr.12
            #   heap2[swap.4] = $274binary_subscr.15  :: (Array(int32, 1, 'C', False, aligned=True), int64, int32) -> none
            #   del $274binary_subscr.15

            heap2[elt], heap2[swap] = heap2[swap], heap2[elt]

            # --- LINE 223 --- 
            #   elt.1 = swap.4  :: int64
            #   del swap.4
            #   elt.2 = elt.1  :: int64

            elt = swap

label 0:
    sched = arg(0, name=sched)               ['sched']
    distances = arg(1, name=distances)       ['distances']
    indices = arg(2, name=indices)           ['indices']
    $4load_global.0 = global(range: <class 'range'>) ['$4load_global.0']
    $const18.3.1 = const(int, 0)             ['$const18.3.1']
    $20binary_subscr.4 = getitem(value=sched, index=$const18.3.1, fn=<built-in function getitem>) ['$20binary_subscr.4', '$const18.3.1', 'sched']
    $const32.6.2 = const(int, 1)             ['$const32.6.2']
    $34binary_subscr.7 = getitem(value=sched, index=$const32.6.2, fn=<built-in function getitem>) ['$34binary_subscr.7', '$const32.6.2', 'sched']
    $44load_global.8 = global(np: <module 'numpy' from '/root/subliminal-learning/.venv/lib/python3.11/site-packages/numpy/__init__.py'>) ['$44load_global.8']
    $56load_method.10 = getattr(value=$44load_global.8, attr=uint8) ['$44load_global.8', '$56load_method.10']
    $const78.11.2 = const(int, 1)            ['$const78.11.2']
    $84call.12 = call $56load_method.10($const78.11.2, func=$56load_method.10, args=[Var($const78.11.2, <string>:2)], kws=(), vararg=None, varkwarg=None, target=None) ['$56load_method.10', '$84call.12', '$const78.11.2']
    $binop_add94.13 = $34binary_subscr.7 + $84call.12 ['$34binary_subscr.7', '$84call.12', '$binop_add94.13']
    $102call.14 = call $4load_global.0($20binary_subscr.4, $binop_add94.13, func=$4load_global.0, args=[Var($20binary_subscr.4, <string>:2), Var($binop_add94.13, <string>:2)], kws=(), vararg=None, varkwarg=None, target=None) ['$102call.14', '$20binary_subscr.4', '$4load_global.0', '$binop_add94.13']
    $112get_iter.15 = getiter(value=$102call.14) ['$102call.14', '$112get_iter.15']
    $phi114.0 = $112get_iter.15              ['$112get_iter.15', '$phi114.0']
    jump 114                                 []
label 114:
    $114for_iter.1 = iternext(value=$phi114.0) ['$114for_iter.1', '$phi114.0']
    $114for_iter.2 = pair_first(value=$114for_iter.1) ['$114for_iter.1', '$114for_iter.2']
    $114for_iter.3 = pair_second(value=$114for_iter.1) ['$114for_iter.1', '$114for_iter.3']
    $phi116.1 = $114for_iter.2               ['$114for_iter.2', '$phi116.1']
    branch $114for_iter.3, 116, 124          ['$114for_iter.3']
label 116:
    parfor__index_78 = $phi116.1             ['$phi116.1', 'parfor__index_78']
    __sentinel__0 = const(int, 0)            ['__sentinel__0']
    jump 114                                 []
label 124:
    $const124.0.0 = const(NoneType, None)    ['$const124.0.0']
    $126return_value.1 = cast(value=$const124.0.0) ['$126return_value.1', '$const124.0.0']
    return $126return_value.1                ['$126return_value.1']
UMAP failed: Failed in nopython mode pipeline (step: nopython frontend)
[1m[1m[1m[1m[1mFailed in nopython mode pipeline (step: nopython frontend)
[1m[1mUntyped global name 'print':[0m [1m[1mCannot determine Numba type of <class 'function'>[0m
[1m
File ".venv/lib/python3.11/site-packages/pynndescent/pynndescent_.py", line 253:[0m
[1mdef nn_descent_internal_low_memory_parallel(
    <source elided>
        if verbose:
[1m            print("\t", n + 1, " / ", n_iters)
[0m            [1m^[0m[0m
[0m
[0m[1mDuring: Pass nopython_type_inference[0m
[0m[1mDuring: resolving callee type: type(CPUDispatcher(<function nn_descent_internal_low_memory_parallel at 0x7fe136404860>))[0m
[0m[1mDuring: typing of call at /root/subliminal-learning/.venv/lib/python3.11/site-packages/pynndescent/pynndescent_.py (359)
[0m
[0m[1mDuring: resolving callee type: type(CPUDispatcher(<function nn_descent_internal_low_memory_parallel at 0x7fe136404860>))[0m
[0m[1mDuring: typing of call at /root/subliminal-learning/.venv/lib/python3.11/site-packages/pynndescent/pynndescent_.py (359)
[0m
[1m
File ".venv/lib/python3.11/site-packages/pynndescent/pynndescent_.py", line 359:[0m
[1mdef nn_descent(
    <source elided>
    if low_memory:
[1m        nn_descent_internal_low_memory_parallel(
[0m        [1m^[0m[0m

[0m[1mDuring: Pass nopython_type_inference[0m

📊 Loading Neutral results...
  Found 281637 numbers
  Subsampled to 250000 numbers for memory efficiency

🔍 Analyzing 250000 numbers for Neutral...
Processed 10000/250000 numbers...
Processed 20000/250000 numbers...
Processed 30000/250000 numbers...
Processed 40000/250000 numbers...
Processed 50000/250000 numbers...
Processed 60000/250000 numbers...
Processed 70000/250000 numbers...
Processed 80000/250000 numbers...
Processed 90000/250000 numbers...
Processed 100000/250000 numbers...
Processed 110000/250000 numbers...
Processed 120000/250000 numbers...
Processed 130000/250000 numbers...
Processed 140000/250000 numbers...
Processed 150000/250000 numbers...
Processed 160000/250000 numbers...
