
import numpy as np
import numpy.typing as npt
import numba
from typing import List, Tuple

_compiled_func = {}

_FASTMATH = False

@numba.njit(fastmath=_FASTMATH, parallel=True)
def _internal_best_match_internal_numba(c, ars, func, nthreads):
    max_output_mt = np.zeros(nthreads)
    output_idx_mt = np.zeros((nthreads, 5), np.int64)
    t_slice_len = (ars[0].shape[0] + nthreads - 1) // nthreads
    for tt in numba.prange(nthreads):
        max_output = 0.0
        output_idx = [0] * 5
        for i1 in range(t_slice_len * tt, min(t_slice_len * (tt + 1), ars[0].shape[0])):
            s1 = ars[0][i1]
            for i2, a2 in enumerate(ars[1]):
                s2 = s1 + a2
                for i3, a3 in enumerate(ars[2]):
                    s3 = s2 + a3
                    for i4, a4 in enumerate(ars[3]):
                        s4 = s3 + a4
                        for i5, a5 in enumerate(ars[4]):
                            a = s4 + a5
                            output = func(c, a)
                            if output > max_output:
                                max_output = output
                                output_idx = [i1, i2, i3, i4, i5]
        max_output_mt[tt] = max_output
        output_idx_mt[tt] = np.array(output_idx)
    best_tt = np.argmax(max_output_mt)
    return max_output_mt[best_tt], output_idx_mt[best_tt]

def best_match_internal_numba(c: npt.NDArray[np.float64], ars: List[npt.NDArray[np.float64]], formula: str) -> Tuple[float, List[int]]:
    formula = formula.format(**{
        "lvl": "c[0]",
        "base_hp": "c[1]",
        "hp": "(c[2] + a[0] + a[1] * c[1] / 100.0)",
        "base_atk": "c[3]",
        "atk": "(c[4] + a[2] + a[3] * c[3] / 100.0)",
        "base_def": "c[5]",
        "def": "(c[6] + a[4] + a[5] * c[5] / 100.0)",
        "er": "(c[7] + a[6])",
        "em": "(c[8] + a[7])",
        "edmg": "(c[9] + a[8])",
        "cr": "(c[10] + a[9])",
        "cdmg": "(c[11] + a[10])",
        "hb": "(c[12] + a[11])",
    })
    if formula in _compiled_func:
        func = _compiled_func[formula]
    else:
        func = numba.njit(eval(f"lambda c, a: ({formula})"), fastmath=_FASTMATH)
        _compiled_func[formula] = func
    return _internal_best_match_internal_numba(c, numba.typed.List(ars), func, 2)
