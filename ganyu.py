#!/usr/bin/env python3

from artifactsim import Character

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
