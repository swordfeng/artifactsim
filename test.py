#!/usr/bin/env python3

from artifactsim import *
from xiangling import Xiangling_TheCatch_SeveredFate

if __name__ == "__main__":
    xiangling = Xiangling_TheCatch_SeveredFate()
    xiangling.artifacts = [
        Artifact(pos=0, main_prop='HP', edmg_match=False, additional_props=['CDMG', 'EM', 'CR', 'ATKP'], additional_props_value=[26.435, 60.6125, 3.11, 5.825], additional_props_count=[4, 3, 1, 1]),
        Artifact(pos=1, main_prop='ATK', edmg_match=False, additional_props=['HP', 'ER', 'CR', 'EM'], additional_props_value=[567.625, 12.95, 11.27375, 23.3125], additional_props_count=[2, 2, 3, 1]),
        Artifact(pos=2, main_prop='ER', edmg_match=False, additional_props=['ATKP', 'CR', 'ATK', 'EM'], additional_props_value=[8.7375, 9.71875, 31.12, 16.318749999999998], additional_props_count=[2, 3, 2, 1]),
        Artifact(pos=3, main_prop='ATKP', edmg_match=False, additional_props=['CR', 'HPP', 'CDMG', 'ER'], additional_props_value=[6.9975, 10.485, 5.4425, 15.54], additional_props_count=[2, 2, 1, 3]),
        Artifact(pos=4, main_prop='CDMG', edmg_match=False, additional_props=['ATKP', 'ER', 'ATK', 'DEF'], additional_props_value=[13.98, 4.5325, 35.01, 41.67], additional_props_count=[3, 1, 2, 2]),
    ]
    print(xiangling.output())