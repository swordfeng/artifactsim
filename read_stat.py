#!/usr/bin/env python3

import pickle
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from artifactsim import *
# from ganyu import Ganyu_Amos_Troupe
from xiangling import Xiangling_TheCatch_SeveredFate

if __name__ == "__main__":
    artifacts_list = []
    with open(sys.argv[1], "rb") as f:
        try:
            while True:
                artifacts_list.append(pickle.load(f))
        except EOFError:
            pass

    c = Xiangling_TheCatch_SeveredFate()

    output_stats = []
    main_stats = defaultdict(lambda: 0)
    main_output_stats = defaultdict(lambda: 0.0)
    max_stats = defaultdict(lambda: 0.0)
    min_stats = defaultdict(lambda: float("inf"))
    grad_stats = defaultdict(lambda: 0)
    crit_stats = defaultdict(lambda: 0.)
    eff_prop_stats = defaultdict(lambda: 0.)
    min_crit_stats = defaultdict(lambda: float("inf"))
    min_eff_prop_stats = defaultdict(lambda: float("inf"))
    single_crit = [0] * 5
    double_crit = [0] * 5
    single_crit_score = [0.0] * 5
    double_crit_score = [0.0] * 5
    
    for artifacts in artifacts_list:
        mains = tuple(a.main_prop for a in artifacts)
        c.artifacts = artifacts
        output = c.output()
        output_stats.append(output)
        main_stats[mains] += 1
        main_output_stats[mains] += output
        max_stats[mains] = max(max_stats[mains], output)
        min_stats[mains] = min(min_stats[mains], output)

    pct10 = np.percentile(output_stats, 10)
    print("PCT10:", pct10)

    for idx, artifacts in enumerate(artifacts_list):
        output = output_stats[idx]
        if output < pct10:
            continue
        mains = tuple(a.main_prop for a in artifacts)
        grad_stats[mains] += 1
        crit = 0.
        eff = 0.
        for a in artifacts:
            if (a.main_prop == CR or CR in a.additional_props) != (a.main_prop == CDMG or CDMG in a.additional_props):
                single_crit[a.pos] += 1
                single_crit_score[a.pos] += get_artifacts_prop([a], CR) * 2 + get_artifacts_prop([a], CDMG)
            if (a.main_prop == CR or CR in a.additional_props) and (a.main_prop == CDMG or CDMG in a.additional_props):
                double_crit[a.pos] += 1
                double_crit_score[a.pos] += get_artifacts_prop([a], CR) * 2 + get_artifacts_prop([a], CDMG)
            for idx, prop in enumerate(a.additional_props):
                if prop in (CR, CDMG):
                    crit += a.additional_props_count[idx]
                if prop in c.eff_props():
                    eff += a.additional_props_count[idx]
                if prop in c.eff_props_small():
                    eff += a.additional_props_count[idx] / 2
        crit_stats[mains] += crit
        min_crit_stats[mains] = min(min_crit_stats[mains], crit)
        eff_prop_stats[mains] += eff
        min_eff_prop_stats[mains] = min(min_eff_prop_stats[mains], eff)

    main_stats = sorted(((v, k) for k, v in main_stats.items()), key=lambda t: -t[0])
    [print(
        f"{k:>6}",
        f"{str(v[2:]):<25}",
        f"{main_output_stats[v] / k:5.1f}",
        f"grad_rate {grad_stats[v]/k*100:5.1f}",
        f"min {min_stats[v]:8.2f}",
        f"max {max_stats[v]:8.2f}",
        f"crit {crit_stats[v] and crit_stats[v]/grad_stats[v]:4.1f}",
        f"eff {eff_prop_stats[v] and eff_prop_stats[v]/grad_stats[v]:4.1f}",
        f"min_crit {min_crit_stats[v]:4.1f}",
        f"min_eff {min_eff_prop_stats[v]:4.1f}",
    ) for k, v in main_stats]
    print(f"Single CRIT: " + str(single_crit))
    print(f"Single CRIT avg score: " + str([s / c for s, c in zip(single_crit_score, single_crit)]))
    print(f"Double CRIT: " + str(double_crit))
    print(f"Double CRIT avg score: " + str([s / c for s, c in zip(double_crit_score, double_crit)]))
    print(f"No CRIT: " + str([9000 - s - d for s, d in zip(single_crit, double_crit)]))
    fig, axis = plt.subplots(2, 1)
    axis[0].set_xlim([min(output_stats), max(output_stats)])
    axis[0].hist(output_stats, bins=100, density=True, cumulative=False)
    axis[0].set_title("Distribution")
    axis[1].set_xlim([min(output_stats), max(output_stats)])
    axis[1].set_ylim([0, 1])
    axis[1].hist(output_stats, bins=100, density=True, cumulative=True)
    axis[1].set_title("CDF")
    plt.show()