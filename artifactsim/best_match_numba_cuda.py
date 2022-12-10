
import numpy as np
import numpy.typing as npt
import numba
from numba import cuda
from typing import List, Tuple

from .constants import ARTIFACT_VEC_MAPPING

_ARTIFACT_SHAPE = len(ARTIFACT_VEC_MAPPING)
_BLOCK_DIM = 256
_compiled_func = {}

_FASTMATH = True

_eval_globals = {
    "where": cuda.jit(lambda cond, x, y: x if cond else y, True),
    "clip": cuda.jit(lambda a, a_min, a_max: min(a_max, max(a, a_min)), True),
    "min": min,
    "max": max,
}

def _gen_kernel_func(eval_func):
    @cuda.jit(fastmath=_FASTMATH)
    def kernel_func(ar0, ar1, ar2t, ar3t, ar4t, max_output_res, output_idx_res):
        s2 = cuda.shared.array(_ARTIFACT_SHAPE, numba.float64)
        s4 = cuda.local.array(_ARTIFACT_SHAPE, numba.float64)
        a = cuda.local.array(_ARTIFACT_SHAPE, numba.float64)
        shared_max_output = cuda.shared.array(_BLOCK_DIM, numba.float64)
        shared_output_idx = cuda.shared.array((5, _BLOCK_DIM), numba.int32)
        threadIdx = cuda.threadIdx.x
        blockIdx = cuda.blockIdx.x
        blockDim = cuda.blockDim.x
        i1 = blockIdx // ar1.shape[0]
        i2 = blockIdx % ar1.shape[0]
        if threadIdx == 0:
            for i in range(_ARTIFACT_SHAPE):
                s2[i] = ar0[i1][i] + ar1[i2][i]
        cuda.syncthreads()
        max_output = 0.0
        output_idx = cuda.local.array(5, numba.int32)
        for it in range(threadIdx, ar2t.shape[1] * ar3t.shape[1], blockDim):
            i3 = it // ar3t.shape[1]
            i4 = it % ar3t.shape[1]
            for i in range(_ARTIFACT_SHAPE):
                s4[i] = s2[i] + ar2t[i][i3] + ar3t[i][i4]
            for i5 in range(ar4t.shape[1]):
                for i in range(_ARTIFACT_SHAPE):
                    a[i] = s4[i] + ar4t[i][i5]
                output = eval_func(a)
                if output > max_output:
                    max_output = output
                    output_idx[0] = i1
                    output_idx[1] = i2
                    output_idx[2] = i3
                    output_idx[3] = i4
                    output_idx[4] = i5
        shared_max_output[threadIdx] = max_output
        for i in range(5):
            shared_output_idx[i][threadIdx] = output_idx[i]
        cuda.syncthreads()
        if threadIdx == 0:
            max_idx = 0
            max_output = shared_max_output[0]
            for i in range(blockDim):
                if shared_max_output[i] > max_output:
                    max_output = shared_max_output[i]
                    max_idx = i
            max_output_res[blockIdx] = max_output
            for i in range(5):
                output_idx_res[i][blockIdx] = shared_output_idx[i][max_idx]
    
    return kernel_func

def best_match_internal(c: npt.NDArray[np.float64], ars: List[npt.NDArray[np.float64]], formula: str) -> Tuple[float, List[int]]:
    formula = formula.format(**{
        "lvl": c[0],
        "base_hp": c[1],
        "hp": f"({c[2]} + a[0] + a[1] * {c[1]} / 100.0)",
        "base_atk": c[3],
        "atk": f"({c[4]} + a[2] + a[3] * {c[3]} / 100.0)",
        "base_def": c[5],
        "def": f"({c[6]} + a[4] + a[5] * {c[5]} / 100.0)",
        "er": f"({c[7]} + a[6])",
        "em": f"({c[8]} + a[7])",
        "edmg": f"({c[9]} + a[8])",
        "cr": f"({c[10]} + a[9])",
        "cdmg": f"({c[11]} + a[10])",
        "hb": f"({c[12]} + a[11])",
    })

    if formula in _compiled_func:
        func = _compiled_func[formula]
    else:
        func = _gen_kernel_func(cuda.jit(eval(f"lambda a: ({formula})", _eval_globals), device=True, fastmath=_FASTMATH))
        _compiled_func[formula] = func

    stream = cuda.stream()
    with stream.auto_synchronize():
        gridDim = ars[0].shape[0] * ars[1].shape[0]
        max_output_res = cuda.device_array(gridDim, np.float64, stream=stream)
        output_idx_res = cuda.device_array((5, gridDim), np.int32, stream=stream)
        ar0 = cuda.to_device(ars[0], stream=stream)
        ar1 = cuda.to_device(ars[1], stream=stream)
        ar2t = cuda.to_device(ars[2].transpose(), stream=stream)
        ar3t = cuda.to_device(ars[3].transpose(), stream=stream)
        ar4t = cuda.to_device(ars[4].transpose(), stream=stream)
        func[gridDim, _BLOCK_DIM, stream](ar0, ar1, ar2t, ar3t, ar4t, max_output_res, output_idx_res)
        max_output_res = max_output_res.copy_to_host(stream=stream)
        output_idx_res = output_idx_res.copy_to_host(stream=stream)
    max_idx = np.argmax(max_output_res)
    return max_output_res[max_idx], output_idx_res[:, max_idx]
