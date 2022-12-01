#!/usr/bin/env python3

import pickle
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

from ganyu import Ganyu_Amos_Troupe

if __name__ == "__main__":
    ganyu = Ganyu_Amos_Troupe()

    main_stats = defaultdict(lambda: 0)
    max_stats = defaultdict(lambda: 0.0)
    min_stats = defaultdict(lambda: 1e8)
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
        except EOFError:
            pass
    main_stats = sorted(((v, k) for k, v in main_stats.items()), key=lambda t: -t[0])
    [print(f"{k:>6} {str(v):<40} min {min_stats[v]:6.2f} max {max_stats[v]:6.2f}") for k, v in main_stats]
    fig, axis = plt.subplots(2, 1)
    axis[0].set_xlim([min(output_stats), max(output_stats)])
    axis[0].hist(output_stats, bins=100, density=True, cumulative=False)
    axis[0].set_title("Distribution")
    axis[1].set_xlim([min(output_stats), max(output_stats)])
    axis[1].set_ylim([0, 1])
    axis[1].hist(output_stats, bins=100, density=True, cumulative=True)
    axis[1].set_title("CDF")
    plt.show()