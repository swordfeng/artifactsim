
import numpy as np
import numpy.typing as npt
import numba
from numba import cuda
from typing import List, Tuple

_ARTIFACT_SHAPE = 12
_BLOCK_DIM = 256
_compiled_func = {}


def _gen_kernel_func(eval_func):
    @cuda.jit
    def kernel_func(ar0, ar1, ar2t, ar3t, ar4t, max_output_res, output_idx_res):
        s2 = cuda.shared.array(_ARTIFACT_SHAPE, numba.float64)
        # ar4tshared = cuda.shared.array((_ARTIFACT_SHAPE, _MAX_N_CIRCLET), numba.float64)
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
        # for j in range(threadIdx, ar4t.shape[1], blockDim):
        #     for i in range(_ARTIFACT_SHAPE):
        #         ar4tshared[i][j] = ar4t[i][j]
        # cuda.syncthreads()
        s2local = cuda.local.array(_ARTIFACT_SHAPE, numba.float64)
        s4 = cuda.local.array(_ARTIFACT_SHAPE, numba.float64)
        a = cuda.local.array(_ARTIFACT_SHAPE, numba.float64)
        max_output = 0.0
        output_idx = cuda.local.array(5, numba.int32)
        for i in range(_ARTIFACT_SHAPE):
            s2local[i] = s2[i]
        for it in range(threadIdx, ar2t.shape[1] * ar3t.shape[1], blockDim):
            i3 = it // ar3t.shape[1]
            i4 = it % ar3t.shape[1]
            for i in range(_ARTIFACT_SHAPE):
                s4[i] = s2local[i] + ar2t[i][i3] + ar3t[i][i4]
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

def best_match_internal_numba(c: npt.NDArray[np.float64], ars: List[npt.NDArray[np.float64]], formula: str) -> Tuple[float, List[int]]:
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

    func = cuda.jit(eval(f"lambda a: ({formula})"), device=True)
    kernel = _gen_kernel_func(func)

    gridDim = ars[0].shape[0] * ars[1].shape[0]
    max_output_res = np.zeros(gridDim, np.float64)
    output_idx_res = np.zeros((5, gridDim), np.int32)
    kernel[gridDim, _BLOCK_DIM](ars[0], ars[1], ars[2].transpose(), ars[3].transpose(), ars[4].transpose(), max_output_res, output_idx_res)
    max_idx = np.argmax(max_output_res)
    return max_output_res[max_idx], output_idx_res[:, max_idx]
