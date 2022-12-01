#!/usr/bin/env python3

import pickle
from collections import defaultdict
from lib import *
from best_match_numba_vec import best_match_internal as bm_vec
from best_match_numba_cuda import best_match_internal as bm_cuda

if __name__ == "__main__":
    ganyu = Ganyu_Amos_Troupe()

    main_stats = defaultdict(lambda: 0)
    max_stats = defaultdict(lambda: 0.0)
    min_stats = defaultdict(lambda: 1e8)
    output_stats = []
    with open("ganyu.pickle", "rb") as f:
        try:
            while True:
                artifacts = pickle.load(f)
                mains = tuple(a.main_prop for a in artifacts)
                ganyu.artifacts = artifacts
                output = ganyu.output()
                main_stats[mains] += 1
                output_stats.append(output)
                max_stats[mains] = max(max_stats[mains], output)
                min_stats[mains] = min(min_stats[mains], output)
        except EOFError:
            pass
    main_stats = sorted(((v, k) for k, v in main_stats.items()), key=lambda t: -t[0])
    [print(f"{k:>6} {str(v):<40} min {min_stats[v]:6.2f} max {max_stats[v]:6.2f}") for k, v in main_stats]