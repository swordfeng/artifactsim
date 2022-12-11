#!/usr/bin/env python3

from artifactsim import Character

class Xiangling_TheCatch_SeveredFate(Character):
    def __init__(self):
        super().__init__(
            lvl=90.0,
            base_hp=10875.0,
            base_atk=735.0,
            base_def=669.0,
            cr=5.0,
            cdmg=50.0,
            er=165.9,
            em=96.0,
            edmg=0.0,
            hb=0.0,
        )
    
    @Character.disp_atk.getter
    def disp_atk(self) -> float:
        # 双火 班尼特原木刀天赋12
        return super().disp_atk + self.base_atk * 0.25 + 846.7
    
    def output_formula(self) -> str:
        # Q 旋火轮 融化
        return """
            ({atk} * 2.38)
            * (1.0 + 0.32 + {er} / 400.0 + 0.15 + {edmg} / 100.0)
            * (1.0 + ({cr} + 12.0) * {cdmg} / 10000.0)
            * (2.0 * (1.0 + (2.78 * {em}) / ({em} + 1400.0)))
            * (({lvl} + 100) / ({lvl} + 100 + (90 + 100)))
            * (1.0 + (0.15 - 0.1) / 2)
            * ({er} > 250.0)
        """
