#!/usr/bin/env python3

import itertools
import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Callable, Union

from .constants import *
from .character import Character
from .artifact import Artifact

def best_match_slow(character: Character, artifacts: List[Artifact]) -> Tuple[float, List[Artifact]]:
    al = [[a for a in artifacts if a.pos == pos] for pos in range(5)]
    al = [a for a in al if al]
    max_output = 0.0
    current_best = []
    for comb in itertools.product(*al):
        character.artifacts = list(comb)
        output = character.output()
        if output > max_output:
            max_output = output
            current_best = comb
    return max_output, current_best


def character2vec(character: Character) -> npt.NDArray[np.float64]:
    c = np.zeros(len(CHARACTER_VEC_MAPPING))
    for idx, ch_prop in enumerate(CHARACTER_VEC_MAPPING):
        if hasattr(character, f"disp_{ch_prop}"):
            c[idx] = getattr(character, f"disp_{ch_prop}")
        else:
            c[idx] = getattr(character, ch_prop)
    return c

def artifact2vec(artifact: Artifact) -> npt.NDArray[np.float64]:
    a = np.zeros(len(ARTIFACT_VEC_MAPPING))
    if artifact.main_prop != EDMG or artifact.edmg_match:
        a[ARTIFACT_VEC_MAPPING.index(artifact.main_prop)] = ARTIFACT_MAIN_PROP_MAX[artifact.main_prop]
    for idx, prop in enumerate(artifact.additional_props):
        a[ARTIFACT_VEC_MAPPING.index(prop)] = artifact.additional_props_value[idx]
    return a

def best_match(character: Character, artifacts: List[Artifact], search_method: Union[str, Callable] = "cpu") -> Tuple[float, List[Artifact]]:
    if search_method == "cpu":
        from .best_match_numba_vec import best_match_internal
        search_method = best_match_internal
    elif search_method == "gpu":
        from .best_match_numba_cuda import best_match_internal
        search_method = best_match_internal
    elif type(search_method) is str:
        raise ValueError(f"Bad search method: {search_method}")
    character.artifacts = []
    c = character2vec(character)
    ars = [[] for _ in range(5)]
    aidx = [[] for _ in range(5)]
    for idx, a in enumerate(artifacts):
        ars[a.pos].append(artifact2vec(a))
        aidx[a.pos].append(idx)
    for i in range(5):
        if not ars[i]:
            ars[i].append(np.zeros(len(ARTIFACT_VEC_MAPPING)))
    ars = [np.array(al) for al in ars]
    max_output, output_idx = search_method(c, ars, character.output_formula())
    output_artifacts = []
    for pos in range(5):
        if aidx[pos]:
            output_artifacts.append(artifacts[aidx[pos][output_idx[pos]]])
    return max_output, output_artifacts
