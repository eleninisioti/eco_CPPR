import yaml
import pickle
import numpy as np
import sys
import os
from scipy.signal import find_peaks
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import f_oneway, tukey_hsd

fig_width = 7
fig_height = 5

def post_process(project):
    trial = 0
    gen = 980
    metric = "sustainability"

    with open(project + "/trial_" + str(trial) + "/eval/data/gen_" + str(gen) + ".pkl", "rb") as f:
        results = pickle.load(f)

    results = results[-1]

    # plot histogram for each agent
    agents = list(set(results["agent_idx"]))
    results_gen = results.loc[results["gen"] == gen]
    signif_result = 0

    for agent_idx in agents:
        results_agent = results_gen.loc[results_gen["agent_idx"] == agent_idx]

        bar_heights = []
        bar_errors = []

        tests = ["test_firstmove_low",
                 "test_firstmove_medium",
                 "test_firstmove_high"

                 ]
        for_significance = []
        for test in tests:
            test_agent = results_agent.loc[results_agent["test_type"]==test]
            sustain = test_agent[metric].tolist()
            sustain = [800 if el ==0 else el for el in sustain ]
            bar_heights.append(np.mean(sustain))
            bar_errors.append(np.std(sustain))
            for_significance.append(sustain)


        F, p = f_oneway(*for_significance)
        if p < 0.05:
            print("significance for agent ", str(agent_idx))
            res = tukey_hsd(*for_significance)
            print(res)
            signif_result += 1

        plt.figure(figsize=(fig_width, fig_height))
        plt.bar(tests, bar_heights)
        plt.title('Agent_'+ str(agent_idx))
        plt.xlabel('Test')
        plt.ylabel(metric)
        plt.errorbar(tests, bar_heights, bar_errors)
        save_dir = project  + "/trial_" + str(trial) + "/eval/media/post/"
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        plt.savefig(save_dir + '/agent_' + str(agent_idx) + "_" + metric +  '.png', dpi=400)
        #plt.show()
        plt.clf()
    print("percentage of significant", signif_result / len(agents))


if __name__ == "__main__":
    n_trials = 1
    top_dir = "projects/" + sys.argv[1]
    project_dirs = [top_dir + "/" + el for el in os.listdir(top_dir) if os.path.isdir(top_dir + "/" + el)]
    for project in project_dirs:
        post_process(project)
