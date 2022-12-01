#!/usr/bin/env python3

import time

from lib import *
from best_match_numba_vec import best_match_internal as bm_vec
from best_match_numba_cuda import best_match_internal as bm_cuda


if __name__ == "__main__":
    # ganyu = Ganyu_Amos_Troupe()
    # ganyu.artifacts = [
    #     Artifact(0, HP, True, [ATK, CR, CDMG, EM], [14.0, 3.5, 26.4, 33.0], []),
    #     Artifact(1, ATK, True, [DEF, CDMG, ATKP, CR], [35.0, 7.0, 19.8, 3.9], []),
    #     Artifact(2, ATKP, True, [CR, ER, EM, HPP], [7.4, 19.4, 42.0, 5.8], []),
    #     Artifact(3, EDMG, True, [CDMG, ATK, ATKP, HP], [22.5, 14.0, 14.6, 239.0], []),
    #     Artifact(4, CR, True, [EM, DEF, ER, CDMG], [37.0, 23.0, 5.2, 36.5], []),
    # ]
    # print(ganyu.output())
    ganyu = Ganyu_Amos_Troupe()

    artifacts = random_artifacts(200)

    tic = time.perf_counter()
    max_output, best_artifacts = best_match_opt(ganyu, artifacts, bm_cuda)
    toc = time.perf_counter()
    # [print(artifact) for artifact in best_artifacts]
    print(max_output)
    print(toc - tic)

    tic = time.perf_counter()
    max_output, best_artifacts = best_match_opt(ganyu, artifacts, bm_vec)
    toc = time.perf_counter()
    print(max_output)
    print(toc - tic)
