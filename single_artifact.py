from artifactsim import *
from ganyu import Ganyu_Amos_Troupe
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ganyu = Ganyu_Amos_Troupe()

    output_stats = []
    for _ in range(10000):
        ganyu.artifacts = [new_artifact(CIRCLET)]
        output_stats.append(ganyu.output())
    
    
    fig, axis = plt.subplots(2, 1)
    axis[0].set_xlim([min(output_stats), max(output_stats)])
    axis[0].hist(output_stats, bins=100, density=True, cumulative=False)
    axis[0].set_title("Distribution")
    axis[1].set_xlim([min(output_stats), max(output_stats)])
    axis[1].set_ylim([0, 1])
    axis[1].hist(output_stats, bins=100, density=True, cumulative=True)
    axis[1].set_title("CDF")
    plt.show()