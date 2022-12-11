#!/usr/bin/env python3

from collections import defaultdict

from .constants import EDMG, HPP, HP, ATKP, ATK, DEFP, DEF
from .character import Character
from .artifact import Artifact

from typing import List

def effective_prop_exists(c: Character, ars: List[Artifact]) -> List[bool]:
    return [
        any(
            (prop == a.main_prop and (prop != EDMG or a.edmg_match)) or (prop in a.additional_props)
            for prop in c.eff_props
        )
        for a in ars
    ]


def no_better_artifact(c: Character, ars: List[Artifact]) -> List[bool]:
    nars = []
    eff_props = c.eff_props
    eff_props_small = c.eff_props_small
    small_to_big = {HP: HPP, ATK: ATKP, DEF: DEFP}
    base_val = {HP: c.base_hp, ATK: c.base_atk, DEF: c.base_def}
    for a in ars:
        main_eff = a.main_prop in eff_props and (a.main_prop != EDMG or a.edmg_match)
        main = a.main_prop if main_eff else None
        add = defaultdict(lambda: 0.0)
        for prop, val in zip(a.additional_props, a.additional_props_value):
            if prop in eff_props:
                add[prop] += val
            elif prop in eff_props_small:
                add[small_to_big[prop]] += val * 100.0 / base_val[prop]
        nars.append((a.pos, main, add))

    def is_better_than(ai, aj) -> bool:
        if not ai[1] and aj[1]:
            return False
        if ai[1] and aj[1] and ai[1] != aj[1]:
            return False
        addi, addj = ai[2], aj[2]
        i_has_larger_prop = False
        for prop in eff_props:
            ai_val = addi[prop]
            aj_val = addj[prop]
            if aj_val > ai_val:
                return False
            if aj_val < ai_val:
                i_has_larger_prop = True
        return (ai[1] and not aj[1]) or i_has_larger_prop

    best_records = [set() for _ in range(5)]
    for i, ai in enumerate(nars):
        better_exists = False
        rm_set = set()
        for idxj in best_records[ai[0]]:
            aj = nars[idxj]
            if is_better_than(aj, ai):
                better_exists = True
                break
            if is_better_than(ai, aj):
                rm_set.add(idxj)
        if not better_exists:
            best_records[ai[0]] -= rm_set
            best_records[ai[0]].add(i)
    rset = set.union(*best_records)
    return [i in rset for i in range(len(ars))]