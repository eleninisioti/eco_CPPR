import yaml
import pickle
import numpy as np
import sys
import os
from scipy.signal import find_peaks

def detect_tragedy(mean_rewards):
    window = 10
    tragedy = 0
    moving_average = np.convolve(mean_rewards, np.ones(window) / window, mode='valid')

    peak = np.max(moving_average)
    peak_loc = [idx for idx, el in enumerate(moving_average) if el==peak]
    offset = peak / 3
    local_minima = 0

    for idx, el in enumerate(moving_average):
        if el < (peak - offset) and el > 5:
            tragedy = idx
    """
    local_minima,_ = find_peaks(mean_rewards, threshold=0.8)
    local_minima = list(local_minima)
    local_minima_duration = 10
    local_minima_discard = []
    local_minima = [el for el in local_minima if el < (len(mean_rewards)-10)]
    for el in local_minima:
        for dur in range(el, el + local_minima_duration):
            if mean_rewards[dur+1] > 0.5 and el not in local_minima_discard:
                local_minima_discard.append(el)

    local_minima = [el for el in local_minima if el not in local_minima_discard]
    return tragedy, local_minima
    """
    local_minima_duration = 10
    local_minima = []
    for idx, el in enumerate(moving_average[int(local_minima_duration/2):-local_minima_duration]):
        if el < 0.01 :
            checksum = np.sum([check for check in range(idx, idx+local_minima_duration) if moving_average[check] > 0.01])
            exists = np.sum([1 for c in range(int(idx-local_minima_duration/2), int(idx + local_minima_duration/2)) if (c in local_minima)])
            if not checksum and not exists:
                local_minima.append(idx)

    return tragedy, local_minima


def post_process(project_dir):
    post_process = {}

    with open(project_dir + "/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        last_gen = config["num_gens"]

    last_gen = 500
    # load training data
    with open(project_dir + "/train/data/gen_" + str(last_gen) + ".pkl", "rb") as f:
        results = pickle.load(f)
        post_process["tragedy_train"], post_process["local_minima"] = detect_tragedy(results["mean_rewards"])

    # load eval data
    with open(project_dir + "/eval/data/gen_" + str(last_gen) + ".pkl", "rb") as f:
        results_eval = pickle.load(f)

        max_foraging = np.max(results_eval["efficiency"]["test_foraging"])
        last_foraging = results_eval["efficiency"]["test_foraging"][-1]

        epsilon = 2*np.std(results_eval["efficiency"]["test_foraging"])
        if last_foraging < (max_foraging - epsilon):
            foraging_degraded = True
        else:
            foraging_degraded = False

        foraging_threshold = 0.2  # an agent consumes a resource at least every 5 steps
        if max_foraging / 15 < foraging_threshold:
            foraging_succeeded = False
        else:
            foraging_succeeded = True

        post_process["eval_foraging"] = {"succeeded": foraging_succeeded,
                                         "degraded": foraging_degraded}

        max_exploration = np.max(results_eval["efficiency"]["test_exploration"])

        if (max_exploration / 15) > 2:
            exploration_succeeded = True
        else:
            exploration_succeeded = False
        post_process["exploration_succeeded"] = exploration_succeeded

        sust_tasks = ["test_sustainability_low", "test_sustainability_high"]
        for task in sust_tasks:
            max_sustainability = np.max(results_eval["sustainability"][task])
            last_sustainability = float(results_eval["sustainability"][task][-1])

            efficiency_max = float(np.max(results_eval["sustainability"][task]))
            efficiency_last = float(results_eval["efficiency"][task][-1])

            epsilon = 2*np.std(results_eval["sustainability"][task])
            if last_sustainability < (max_sustainability - epsilon):
                sustainable_degraded = True
            else:
                sustainable_degraded = False

            sustainability_threshold = 1000 * 0.9  # an agent consumes a resource at least every 5 steps
            if max_sustainability < sustainability_threshold:
                sustainable_once = False
            else:
                sustainable_once = True

            post_process["eval_" + task] = {"sustainable_once": sustainable_once,
                                            "sustainable_degraded": sustainable_degraded,
                                            "efficiency_max": efficiency_max,
                                            "efficiency_last": efficiency_last
                                            }

    with open(project_dir + '/process.yaml', "w") as f:
        yaml.dump(post_process, f)

if __name__ == "__main__":
    top_dir = "projects/" + sys.argv[1]
    project_dirs = [top_dir + "/" + el for el in os.listdir(top_dir) if os.path.isdir(top_dir + "/" + el)]
    for project in project_dirs:
        post_process(project)