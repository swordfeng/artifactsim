#!/usr/bin/env python3

from dataclasses import dataclass
import random
from typing import List, Optional

from .constants import *

@dataclass
class Artifact:
    pos: int
    main_prop: str
    edmg_match: bool
    additional_props: List[str]
    additional_props_value: List[float]
    additional_props_count: List[int]


def new_artifact(pos: Optional[int] = None) -> Artifact:
    if pos is None:
        pos = random.randrange(5)
    main_prop = random.choices(list(ARTIFACT_MAIN_PROP_DIST[pos].keys()), list(ARTIFACT_MAIN_PROP_DIST[pos].values()))[0]
    artifact = Artifact(
        pos=pos,
        main_prop=main_prop,
        edmg_match=(random.randrange(8) == 0),
        additional_props=[],
        additional_props_value=[],
        additional_props_count=[],
    )
    add_prop(artifact)
    add_prop(artifact)
    add_prop(artifact)
    if random.randrange(5) == 0:
        add_prop(artifact)
    return artifact

def random_artifacts(n: int = 1) -> List[Artifact]:
    res = []
    for _ in range(n):
        artifact = new_artifact()
        for __ in range(5):
            add_prop(artifact)
        res.append(artifact)
    return res

def add_prop(artifact: Artifact) -> None:
    if len(artifact.additional_props) < 4:
        dist = {
            prop: weight
            for prop, weight in ARTIFACT_ADDITIONAL_PROP_DIST.items()
            if prop != artifact.main_prop and prop not in artifact.additional_props
        }
        prop = random.choices(list(dist.keys()), list(dist.values()))[0]
        value = ADDITIONAL_PROP_BASE[prop] * random.choice(ADDITIONAL_PROP_LEVELS)
        artifact.additional_props.append(prop)
        artifact.additional_props_value.append(value)
        artifact.additional_props_count.append(1)
    else:
        idx = random.randrange(len(artifact.additional_props))
        prop = artifact.additional_props[idx]
        value = ADDITIONAL_PROP_BASE[prop] * random.choice(ADDITIONAL_PROP_LEVELS)
        artifact.additional_props_value[idx] += value
        artifact.additional_props_count[idx] += 1

def get_artifacts_prop(artifacts: List[Artifact], prop: str) -> float:
    res = 0.0
    for artifact in artifacts:
        if artifact.main_prop == prop:
            if prop != EDMG or artifact.edmg_match:
                res += ARTIFACT_MAIN_PROP_MAX[prop]
        try:
            res += artifact.additional_props_value[artifact.additional_props.index(prop)]
        except ValueError:
            pass
    return res
