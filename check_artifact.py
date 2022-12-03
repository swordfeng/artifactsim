#!/usr/bin/env python3

import pickle
import sys
import random
import multiprocessing as mp
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from artifactsim import *
from ganyu import Ganyu_Amos_Troupe

ganyu = Ganyu_Amos_Troupe()

def gen_artifact():
    artifact = Artifact(1, ATK, True, [CDMG, HP, DEFP, CR], [7.8, 209.0, 5.1, 2.7], [1, 1, 1, 1]) #.62599
    # artifact = Artifact(1, ATK, True, [DEF, CR, EM, ER], [20.84, 3.89, 18.65, 5.18], [1, 1, 1, 1]) #.59222
    # artifact = Artifact(1, ATK, True, [CR, HPP, ATKP], [3.1, 4.1, 5.3], [1, 1, 1]) #.53516
    # artifact = Artifact(1, ATK, True, [DEF, ATKP, DEFP, EM], [20.84, 5.1, 5.3, 18.65], [1, 1, 1, 1]) #.45236
    # artifact = Artifact(1, ATK, True, [DEF, CR, ATK, ER], [20.84, 3.89, 18.65, 5.18], [1, 1, 1, 1]) #.37179
    # artifact = Artifact(1, ATK, True, [DEF, DEFP, ER, HP], [20.84, 5.1, 5.1, 209.0], [1, 1, 1, 1]) #.0
    for _ in range(5):
        add_artifact_prop(artifact)
    return artifact

def task(artifacts_list, pct10):
    better_count = 0
    for test_artifact in [gen_artifact() for _ in range(100)]:
        for artifacts in random.sample(artifacts_list, k=100):
            ganyu.artifacts = artifacts
            old_output = ganyu.output()
            d = {artifact.pos: artifact for artifact in artifacts}
            d[test_artifact.pos] = test_artifact
            ganyu.artifacts = list(d.values())
            new_output = ganyu.output()
            if new_output > old_output * 0.9 and new_output > pct10:
                better_count += 1
    return better_count

def eval_output(artifacts):
    ganyu.artifacts = artifacts
    return ganyu.output()

if __name__ == "__main__":
    artifacts_list = []
    with open(sys.argv[1], "rb") as f:
        try:
            while True:
                artifacts_list.append(pickle.load(f))
        except EOFError:
            pass

    pct10 = np.percentile([eval_output(a) for a in artifacts_list], 10)
    with mp.Pool(16) as pool:
        better_count = sum(pool.starmap(task, [(artifacts_list, pct10)] * 50))

    print(better_count / (100 * 100 * 50))
