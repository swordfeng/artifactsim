#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import List, Sequence

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

    def eff_props(self) -> Sequence[str]:
        if not hasattr(self, "_eff_props"):
            props = set()
            formula = self.output_formula().format(**{
                "lvl": 90,
                "base_hp": 10000.0,
                "hp": "(add('HPP') or 0.0)",
                "base_atk": 200.0,
                "atk": "(add('ATKP') or 0.0)",
                "base_def": 200.0,
                "def": "(add('DEFP') or 0.0)",
                "er": "(add('ER') or 0.0)",
                "em": "(add('EM') or 0.0)",
                "edmg": "(add('EDMG') or 0.0)",
                "cr": "(add('CR') or 0.0)",
                "cdmg": "(add('CDMG') or 0.0)",
                "hb": "(add('HB') or 0.0)",
            })
            eval(f"lambda add: ({formula})")(props.add)
            self._eff_props = tuple(props)
        return self._eff_props
    
    def eff_props_small(self) -> Sequence[str]:
        mapping = {HPP: HP, ATKP: ATK, DEFP: DEF}
        return tuple(
            mapping[prop]
            for prop in self.eff_props()
            if prop in mapping
        )