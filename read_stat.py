#!/usr/bin/env python3

import pickle
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

from artifactsim import CR, CDMG, get_artifacts_prop
from ganyu import Ganyu_Amos_Troupe

if __name__ == "__main__":
    ganyu = Ganyu_Amos_Troupe()

    main_stats = defaultdict(lambda: 0)
    max_stats = defaultdict(lambda: 0.0)
    min_stats = defaultdict(lambda: 1e8)
    single_crit = [0] * 5
    double_crit = [0] * 5
    single_crit_score = [0.0] * 5
    double_crit_score = [0.0] * 5
    output_stats = []
    with open(sys.argv[1], "rb") as f:
        try:
            while True:
                artifacts = pickle.load(f)
                mains = tuple(a.main_prop for a in artifacts)
                ganyu.artifacts = artifacts
                output = ganyu.output()
                main_stats[mains] += 1
                output_stats.append(output)
                max_stats[mains] = max(max_stats[mains], output)
                min_stats[mains] = min(min_stats[mains], output)
                for a in artifacts:
                    if (a.main_prop == CR or CR in a.additional_props) != (a.main_prop == CDMG or CDMG in a.additional_props):
                        single_crit[a.pos] += 1
                        single_crit_score[a.pos] += get_artifacts_prop([a], CR) * 2 + get_artifacts_prop([a], CDMG)
                    if (a.main_prop == CR or CR in a.additional_props) and (a.main_prop == CDMG or CDMG in a.additional_props):
                        double_crit[a.pos] += 1
                        double_crit_score[a.pos] += get_artifacts_prop([a], CR) * 2 + get_artifacts_prop([a], CDMG)
        except EOFError:
            pass
    main_stats = sorted(((v, k) for k, v in main_stats.items()), key=lambda t: -t[0])
    [print(f"{k:>6} {str(v):<40} min {min_stats[v]:6.2f} max {max_stats[v]:6.2f}") for k, v in main_stats]
    print(f"Single CRIT: " + str(single_crit))
    print(f"Single CRIT avg score: " + str([s / c for s, c in zip(single_crit_score, single_crit)]))
    print(f"Double CRIT: " + str(double_crit))
    print(f"Double CRIT avg score: " + str([s / c for s, c in zip(double_crit_score, double_crit)]))
    print(f"No CRIT: " + str([10000 - s - d for s, d in zip(single_crit, double_crit)]))
    fig, axis = plt.subplots(2, 1)
    axis[0].set_xlim([min(output_stats), max(output_stats)])
    axis[0].hist(output_stats, bins=100, density=True, cumulative=False)
    axis[0].set_title("Distribution")
    axis[1].set_xlim([min(output_stats), max(output_stats)])
    axis[1].set_ylim([0, 1])
    axis[1].hist(output_stats, bins=100, density=True, cumulative=True)
    axis[1].set_title("CDF")
    plt.show()