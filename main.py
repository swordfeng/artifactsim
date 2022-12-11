#!/usr/bin/env python3

import time

from artifactsim import *
from ganyu import Ganyu_Amos_Troupe

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

    for _ in range(10):
        artifacts = random_artifacts(500)
        artifacts = [a for a, f in zip(artifacts, filters.no_better_artifact(ganyu, artifacts)) if f]
        
        tic = time.perf_counter()
        max_output, best_artifacts = best_match(ganyu, artifacts, "gpu")
        toc = time.perf_counter()
        [print(artifact) for artifact in best_artifacts]
        print(max_output, toc - tic)

        tic = time.perf_counter()
        max_output, best_artifacts = best_match(ganyu, artifacts, "cpu")
        toc = time.perf_counter()
        print(max_output, toc - tic)

        print()
