#!/usr/bin/env python3

import sys
import time
import pickle
from lib import *
from best_match_numba_cuda import best_match_internal as bm_cuda

if __name__ == "__main__":
    ganyu = Ganyu_Amos_Troupe()

    with open(sys.argv[1], "wb") as f:
        for i in range(10000):
            artifacts = random_artifacts(200)

            tic = time.perf_counter()
            max_output, best_artifacts = best_match_opt(ganyu, artifacts, bm_cuda)
            toc = time.perf_counter()
            pickle.dump(best_artifacts, f)
            print(i, max_output, toc - tic)
