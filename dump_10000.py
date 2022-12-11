#!/usr/bin/env python3

import sys
import time
import pickle

from artifactsim import *
from ganyu import Ganyu_Amos_Troupe
from xiangling import Xiangling_TheCatch_SeveredFate
from keqing import Keqing_JadeCutter_ThunderingFury

CPU_THREADS = 16
GPU_THREADS = 4

ARTIFACT_COUNT = 200
BEST_SET_COUNT = 10000

if __name__ == "__main__":
    c = Ganyu_Amos_Troupe()

    with open(sys.argv[1], "wb") as f:
        pickle.dump(type(c), f)

        with BestMatchGenerator(c, ARTIFACT_COUNT, CPU_THREADS, GPU_THREADS) as gen:
            tic = time.perf_counter()
            for i in range(BEST_SET_COUNT):
                max_output, best_artifacts, time_elapsed = next(gen)
                pickle.dump(best_artifacts, f)
                print(i, max_output, time_elapsed)
            toc = time.perf_counter()
    print("Total time:", toc - tic)
