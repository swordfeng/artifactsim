#!/usr/bin/env python3

from dataclasses import dataclass, field
import itertools
import numpy as np
import numpy.typing as npt
import random
from typing import List, Optional, Tuple, Callable


# Positions
FLOWER = 0
PLUME = 1
SANDS = 2
GOBLET = 3
CIRCLET = 4

# Properties
HP = "HP"
HPP = "HPP"
ATK = "ATK"
ATKP = "ATKP"
DEF = "DEF"
DEFP = "DEFP"
ER = "ER"
EM = "EM"
CR = "CR"
CDMG = "CDMG"
EDMG = "EDMG"
HB = "HB"

# Elements
# PHYSICAL
# PYRO
# HYDRO
# DENDRO
# ELECTRO
# ANEMO
# CYRO
# GEO

@dataclass
class Artifact:
    pos: int
    main_prop: str
    edmg_match: bool
    additional_props: List[str]
    additional_props_value: List[float]
    additional_props_count: List[int]

# https://nga.178.com/read.php?tid=25247146
ARTIFACT_MAIN_PROP_DIST = {
    FLOWER: {HP: 10000},
    PLUME: {ATK: 10000},
    SANDS: {HPP: 2668, ATKP: 2666, DEFP: 2666, ER: 1000, EM: 1000},
    GOBLET: {HPP: 1950, ATKP: 1950, DEFP: 1850, EDMG: 4000, EM: 250},
    CIRCLET: {HPP: 2200, ATKP: 2200, DEFP: 2200, CR: 1000, CDMG: 1000, HB: 1000, EM: 400},
}
ARTIFACT_MAIN_PROP_MAX = {
    HP: 4780.0,
    ATK: 311.2,
    HPP: 46.6,
    ATKP: 46.6,
    DEFP: 58.3,
    ER: 51.8,
    EM: 186.5,
    CR: 31.1,
    CDMG: 62.2,
    EDMG: 46.6,  # Physical is 58.3
    HB: 35.9,
}

ARTIFACT_ADDITIONAL_PROP_DIST = {HP: 1500, ATK: 1500, DEF: 1500, HPP: 1000, ATKP: 1000, DEFP: 1000, ER: 1000, EM: 1000, CR: 750, CDMG: 750}

# https://nga.178.com/read.php?tid=31774495
ADDITIONAL_PROP_BASE = {
    HP: 239.0,
    HPP: 4.66,
    ATK: 15.56,
    ATKP: 4.66,
    DEF: 18.52,
    DEFP: 5.83,
    ER: 5.18,
    EM: 18.65,
    CR: 3.11,
    CDMG: 6.22,
}
ADDITIONAL_PROP_LEVELS = [1.25, 1.125, 1.0, 0.875]

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

@dataclass
class Character:
    lvl: int
    base_hp: float
    base_atk: float
    base_def: float
    cr: float
    cdmg: float
    er: float
    em: float
    edmg: float
    hb: float
    artifacts: List[Artifact] = field(default_factory=list)

    def set_artifacts(self, artifacts: List[Artifact]) -> None:
        self.artifacts = list(artifacts)

    @property
    def disp_hp(self) -> float:
        return self.base_hp + get_artifacts_prop(self.artifacts, HP) + self.base_hp * get_artifacts_prop(self.artifacts, HPP) / 100.0

    @property
    def disp_atk(self) -> float:
        return self.base_atk + get_artifacts_prop(self.artifacts, ATK) + self.base_atk * get_artifacts_prop(self.artifacts, ATKP) / 100.0

    @property
    def disp_def(self) -> float:
        return self.base_def + get_artifacts_prop(self.artifacts, DEF) + self.base_def * get_artifacts_prop(self.artifacts, DEFP) / 100.0

    @property
    def disp_er(self) -> float:
        return self.er + get_artifacts_prop(self.artifacts, ER)

    @property
    def disp_em(self) -> float:
        return self.em + get_artifacts_prop(self.artifacts, EM)
    
    @property
    def disp_edmg(self) -> float:
        return self.edmg + get_artifacts_prop(self.artifacts, EDMG)

    @property
    def disp_cr(self) -> float:
        return self.cr + get_artifacts_prop(self.artifacts, CR)

    @property
    def disp_cdmg(self) -> float:
        return self.cdmg + get_artifacts_prop(self.artifacts, CDMG)

    @property
    def disp_hb(self) -> float:
        return self.hb + get_artifacts_prop(self.artifacts, HB)

    def output_formula(self) -> str:
        raise NotImplemented
    
    def output(self) -> float:
        if not hasattr(self, "_output_func"):
            formula = self.output_formula().format(**{
                "lvl": "c.lvl",
                "base_hp": "c.base_hp",
                "hp": "c.disp_hp",
                "base_atk": "c.base_atk",
                "atk": "c.disp_atk",
                "base_def": "c.base_def",
                "def": "c.disp_def",
                "er": "c.disp_er",
                "em": "c.disp_em",
                "edmg": "c.disp_edmg",
                "cr": "c.disp_cr",
                "cdmg": "c.disp_cdmg",
                "hb": "c.disp_hb",
            })
            self._output_func = eval("lambda c: (" + formula + ")")
        return self._output_func(self)

class Ganyu_Amos_Troupe(Character):
    def __init__(self):
        super().__init__(
            lvl=90.0,
            base_hp=9797.0,
            base_atk=943.0,
            base_def=630.0,
            cr=5.0,
            cdmg=88.4,
            er=100.0,
            em=80.0,
            edmg=0.0,
            hb=0.0,
        )

    @Character.disp_atk.getter
    def disp_atk(self) -> float:
        return super().disp_atk + self.base_atk * 49.6 / 100.0 + self.base_atk * 0.25
    
    def output_formula(self) -> str:
        return """
            ({atk} * 3.92)
            * (1.0 + 0.12 + 0.08 * 3 + 0.35 + {edmg} / 100.0)
            * (1.0 + ({cr} + 20.0) * {cdmg} / 10000.0)
            * (1.5 * (1.0 + (2.78 * {em}) / ({em} + 1400.0)))
            * (({lvl} + 100) / ({lvl} + 100 + (90 + 100)))
            * (1.0 - 0.1)
        """


def best_match(character: Character, artifacts: List[Artifact]) -> Tuple[float, List[Artifact]]:
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

ch_mapping = ["lvl","base_hp","hp","base_atk","atk","base_def","def","er","em","edmg","cr","cdmg","hb"]
ar_mapping = [HP,HPP,ATK,ATKP,DEF,DEFP,ER,EM,EDMG,CR,CDMG,HB]
def character2np(character: Character) -> npt.NDArray[np.float64]:
    c = np.zeros(len(ch_mapping))
    for idx, ch_prop in enumerate(ch_mapping):
        if hasattr(character, f"disp_{ch_prop}"):
            c[idx] = getattr(character, f"disp_{ch_prop}")
        else:
            c[idx] = getattr(character, ch_prop)
    return c
def artifact2np(artifact: Artifact) -> npt.NDArray[np.float64]:
    a = np.zeros(len(ar_mapping))
    if artifact.main_prop != EDMG or artifact.edmg_match:
        a[ar_mapping.index(artifact.main_prop)] = ARTIFACT_MAIN_PROP_MAX[artifact.main_prop]
    for idx, prop in enumerate(artifact.additional_props):
        a[ar_mapping.index(prop)] = artifact.additional_props_value[idx]
    return a

def best_match_opt(character: Character, artifacts: List[Artifact], internal_func: Callable) -> Tuple[float, List[Artifact]]:
    character.artifacts = []
    c = character2np(character)
    ars = [[] for _ in range(5)]
    aidx = [[] for _ in range(5)]
    for idx, a in enumerate(artifacts):
        ars[a.pos].append(artifact2np(a))
        aidx[a.pos].append(idx)
    for i in range(5):
        if not ars[i]:
            ars[i].append(np.zeros(len(ar_mapping)))
    ars = [np.array(al) for al in ars]
    max_output, output_idx = internal_func(c, ars, character.output_formula())
    output_artifacts = []
    for pos in range(5):
        if aidx[pos]:
            output_artifacts.append(artifacts[aidx[pos][output_idx[pos]]])
    return max_output, output_artifacts

