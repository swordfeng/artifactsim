#!/usr/bin/env python3

import asyncio
import pickle
import sys
import random
import multiprocessing as mp
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from typing import List

from artifactsim import *
from ganyu import Ganyu_Amos_Troupe


def score(c: Character, artifacts_list: List[List[Artifact]], test_artifact: Artifact) -> float:
    old_outputs = []
    new_outputs = []
    for artifacts in artifacts_list:
        c.artifacts = artifacts
        old_output = c.output()
        old_outputs.append(old_output)
        ars = [a for a in artifacts if a.pos != test_artifact.pos]
        ars.append(test_artifact)
        c.artifacts = ars
        new_output = c.output()
        if new_output > old_output * 0.9:
            new_outputs.append(new_output)
    pct10 = np.percentile(old_outputs, 10)
    return sum(1 for o in new_outputs if o > pct10) / len(artifacts_list)

def explore_artifact(c: Character, artifacts_list: List[List[Artifact]], artifact: Artifact, times: int):
    stateset = set()
    if len(artifact.additional_props) < 4 and times >= 1: # == 3
        for prop in ADDITIONAL_PROP_BASE:
            if prop != artifact.main_prop and prop not in artifact.additional_props:
                stateset.add((0, 0, 0, 1, prop))
        times -= 1
    else:
        stateset.add((0,0,0,0))
    laststate = stateset
    for _ in range(times):
        newstate = set()
        for st in laststate:
            for i in range(4):
                newstate.add(tuple(v + 1 if i == j else v for j, v in enumerate(st)))
        laststate = newstate
        stateset |= laststate
    k = list(laststate)
    test_as = []
    for st in k:
        a = deepcopy(artifact)
        a.additional_props += st[4:]
        a.additional_props_value += [0. for _ in st[4:]]
        a.additional_props_count += [0 for _ in st[4:]]
        for i in range(4):
            a.additional_props_value[i] += st[i] * ADDITIONAL_PROP_BASE[a.additional_props[i]] * 1.0625
            a.additional_props_count[i] += st[i]
        test_as.append(a)

    with mp.Pool() as pool:
        v = pool.starmap(score, ((c, artifacts_list, a) for a in test_as))
    
    res = dict(zip(k, v))

    def dfs(st):
        try:
            return res[st]
        except KeyError:
            pass
        p = sum(
            dfs(tuple(v + 1 if i == j else v for j, v in enumerate(st)))
            for i in range(4)
        ) / 4
        res[st] = p
        return p

    for st in stateset:
        dfs(st)
    if len(artifact.additional_props) < 4:
        p0 = 0.0
        total_weight = 0.0
        for prop in ADDITIONAL_PROP_BASE:
            if prop != artifact.main_prop and prop not in artifact.additional_props:
                p0 += res[0, 0, 0, 1, prop] * ARTIFACT_ADDITIONAL_PROP_DIST[prop]
                total_weight += ARTIFACT_ADDITIONAL_PROP_DIST[prop]
        res[0, 0, 0, 0] = p0 / total_weight
    return res


if __name__ == "__main__":
    artifacts_list = []
    with open(sys.argv[1], "rb") as f:
        try:
            while True:
                artifacts_list.append(pickle.load(f))
        except EOFError:
            pass

    ganyu = Ganyu_Amos_Troupe()
    
    artifact = Artifact(3, EDMG, True, [CDMG, CR, EM, ATKP], [6.22, 3.11, 18.65, 4.66], [1, 1, 1, 1])
    # artifact = Artifact(1, ATK, True, [CDMG, HP, DEFP, CR], [7.8, 209.0, 5.1, 2.7], [1, 1, 1, 1])
    # artifact = Artifact(1, ATK, True, [DEF, CDMG, EM, ER], [20.84, 6.6, 18.65, 5.18], [1, 1, 1, 1])
    # artifact = Artifact(1, ATK, True, [DEF, CR, EM, ER], [20.84, 3.89, 18.65, 5.18], [1, 1, 1, 1])
    # artifact = Artifact(1, ATK, True, [CR, HPP, ATKP], [3.1, 4.1, 5.3], [1, 1, 1])
    # artifact = Artifact(1, ATK, True, [DEF, ATKP, DEFP, EM], [20.84, 5.1, 5.3, 18.65], [1, 1, 1, 1])
    # artifact = Artifact(1, ATK, True, [DEF, CR, ATK, ER], [20.84, 3.89, 18.65, 5.18], [1, 1, 1, 1])
    # artifact = Artifact(1, ATK, True, [DEF, DEFP, ER, HP], [20.84, 5.1, 5.1, 209.0], [1, 1, 1, 1])

    for st, p in sorted(explore_artifact(ganyu, artifacts_list, artifact, 5).items(), key=lambda t: (sum(t[0][:4]), -t[1]) + t[0][4:] + tuple(-x for x in t[0][:4])):
        print(st, p)