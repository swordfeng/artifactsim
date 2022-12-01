#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import List

from .artifact import Artifact, get_artifacts_prop
from .constants import *

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
