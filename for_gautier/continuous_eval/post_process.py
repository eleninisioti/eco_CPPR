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

fig_width = 7
fig_height = 5

def post_process(project):
    trial = 0
    gen = 980

    with open(project + "/trial_" + str(trial) + "/eval/data/gen_" + str(gen) + ".pkl", "rb") as f:
        results = pickle.load(f)

    results = results[-1]

    # plot histogram for each agent
    agents = list(set(results["agent_idx"]))
    for agent_idx in agents:
        results_agent = results.loc[results[agent_idx] == agent_idx]

        bar_heights = []
        bar_errors = []

        tests = ["test_firstmove_low",
                 "test_firstmove_high",
                 "test_firstmove_medium"
                 ]

        for test in tests:
            test_agent = results_agent.loc[results_agent["test"]==test]
            sustain = test_agent["sustainability"].tolist()
            bar_heights.append(np.mean(sustain))
            bar_errors.append(np.var(sustain))

        plt.figure(figsize=(fig_width, fig_height))
        plt.bar(tests, bar_heights)
        plt.title('Agent', str(agent_idx))
        plt.xlabel('Test')
        plt.ylabel('Sustainability %')
        plt.errorbar(tests, bar_heights, bar_errors)
        save_dir = project  + "/eval/media/post/"
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        plt.savefig(save_dir + '/agent_' + str(agent_idx) + '.png', dpi=400)
        plt.show()
        plt.clf()



if __name__ == "__main__":
    n_trials = 1
    top_dir = "projects/" + sys.argv[1]
    project_dirs = [top_dir + "/" + el for el in os.listdir(top_dir) if os.path.isdir(top_dir + "/" + el)]
    for project in project_dirs:
        post_process(project)
