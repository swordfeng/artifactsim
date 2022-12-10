#!/usr/bin/env python3

from artifactsim import Character

class Keqing_JadeCutter_ThunderingFury(Character):
    def __init__(self):
        super().__init__(
            lvl=90.0,
            base_hp=13103.0,
            base_atk=864.72,
            base_def=799.3,
            cr=49.1,
            cdmg=88.4,
            er=0.0,
            em=0.0,
            edmg=15.0,
            hb=0.0,
        )
    
    @Character.disp_hp.getter
    def disp_hp(self) -> float:
        return super().disp_hp + self.base_hp * 0.2
    
    def output_formula(self) -> str:
        # AZ 两段 激化
        return """
            (
                ({atk} + {hp} * 0.012) * 3.2181
                + 1446.85 * 1.15 * (1.0 + (5.0 * {em}) / ({em} + 1200.0) + 0.2)
            )
            * (1.0 + {edmg} / 100.0)
            * (1.0 + {cr} * {cdmg} / 10000.0)
            * (({lvl} + 100) / ({lvl} + 100 + (90 + 100)))
            * (1.0 - 0.1)
        """
