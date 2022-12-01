import numpy as np
import numpy.typing as npt
from typing import List, Tuple

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
    eval_func = eval(f"lambda a: ({formula})")
    max_output = 0.0
    output_idx = [0] * 5
    for i1, a1 in enumerate(ars[0]):
        s1 = a1
        for i2, a2 in enumerate(ars[1]):
            s2 = s1 + a2
            for i3, a3 in enumerate(ars[2]):
                s3 = s2 + a3
                for i4, a4 in enumerate(ars[3]):
                    s4 = s3 + a4
                    a = (ars[4] + s4).transpose()
                    output = eval_func(a)
                    i5 = np.argmax(output)
                    if output[i5] > max_output:
                        max_output = output[i5]
                        output_idx = [i1, i2, i3, i4, i5]
    return max_output, output_idx
