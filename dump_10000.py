#!/usr/bin/env python3

import sys
import time
import pickle
import queue
import threading
import multiprocessing
import numba.cuda
from lib import *
from best_match_numba_vec import best_match_internal as bm_vec
from best_match_numba_cuda import best_match_internal as bm_cuda

CPU_THREADS = 24
GPU_THREADS = 2

ARTIFACT_COUNT = 200
BEST_SET_COUNT = 10000

def task(c, bm_func, output_queue):
    try:
        while True:
            artifacts = random_artifacts(ARTIFACT_COUNT)
            tic = time.perf_counter()
            max_output, best_artifacts = best_match_opt(c, artifacts, bm_func)
            toc = time.perf_counter()
            output_queue.put((max_output, best_artifacts, toc - tic))
    except:
        pass

if __name__ == "__main__":
    ganyu = Ganyu_Amos_Troupe()
    q = multiprocessing.Queue(maxsize=64)
    numba.cuda.synchronize()

    for _ in range(CPU_THREADS):
        t = multiprocessing.Process(target=task, args=(ganyu, bm_vec, q), daemon=True)
        t.start()
    for _ in range(GPU_THREADS):
        t = threading.Thread(target=task, args=(ganyu, bm_cuda, q), daemon=True)
        t.start()

    with open(sys.argv[1], "wb") as f:
        tic = time.perf_counter()
        for i in range(BEST_SET_COUNT):
            max_output, best_artifacts, time_elapsed = q.get()
            pickle.dump(best_artifacts, f)
            print(i, max_output, time_elapsed)
        toc = time.perf_counter()
    print("Total time:", toc - tic)
