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

def compare_single_parameter(top_dir, param_compare):
    parameters = ["gen_length", "nb_agents", "grid_length", "niches_scale", "regrowth_scale"]

    # load all configs
    configs = {}
    project_dirs = [top_dir + "/" + el for el in os.listdir(top_dir) if
                    os.path.isdir(top_dir + "/" + el) and "parametric" not in el]
    for project in project_dirs:
        with open(project + "/trial_0/config.yaml", "r") as f:
            configs[project] = yaml.safe_load(f)

    other_params = [el for el in parameters if el != param_compare]
    all_params = {}
    for parameter in other_params:
        param_values = []
        for project, config in configs.items():
            param_values.append(config[parameter])
        param_values = list(set(param_values))
        all_params[parameter] = param_values

    keys, values = zip(*all_params.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    for case in permutations_dicts:
        case_dir = top_dir + "/parametric_single_" + param_compare + "/"

        for key, value in case.items():
            case_dir += key + "_" + str(value)

        if not os.path.exists(case_dir):
            os.makedirs(case_dir)

        case_projects = list(configs.keys())



        for project, config in configs.items():
            for param, value in case.items():
                if value != config[param]:
                    case_projects.remove(project)
                    break

        with open(case_dir + "/projects.txt", "w") as f:

            for el in case_projects:
                f.write(el + "\n")

        # for each case project load and plot training results
        results = {}
        for case_project in case_projects:
            # last_gen = config[case_project]["num_gens"]
            last_gen = 1400
            ntrials = 3
            eval_freq = 50
            total_frame = []
            for trial in range(ntrials):
                trial_project = case_project + "/trial_" + str(trial)
                with open(trial_project + "/train/data/gen_" + str(last_gen) + ".pkl", "rb") as f:
                    trial_results = pickle.load(f)
                    # make dataframe
                    rewards = [float(el) for el in trial_results["max_rewards"]]
                    gens = list(range(len(rewards)))
                    trial = [trial] * len(rewards)

                    trial_frame = pd.DataFrame({'max_rewards': rewards,
                                                'gen': gens,
                                                'trial': trial
                                                })
                    total_frame.append(trial_frame)
            total_frame = pd.concat(total_frame)

            case_label = param_compare + "_" + str(configs[case_project][param_compare])
            results[case_label] = total_frame

        for case, result in results.items():
            sns.lineplot(data=result,x="gen", y="max_rewards", label=case)
        plt.savefig(case_dir + "/max_rewards.png")
        plt.clf()

        # for each case project load and plot evaluation results
        """
        eval_results = {}
        for case_project in case_projects:
            #last_gen = config[case_project]["num_gens"]
            with open(case_project + "/eval/data/gen_" + str(last_gen) + ".pkl", "wb") as f:
                eval_results[case] = pickle.load(f)

        metrics = ["efficiency", "sustainability"]
        tasks = ["test_foraging", "test_exploration", "test_sustainability_low", "test_sustainability_high"]
        for metric in metrics:
            for task in tasks:
                for case, result in results.items():
                    plt.plot(results[metrics], label=case)
                plt.savefig(case_dir + "/task_" + task + "_metric_" + metric + ".png")
                plt.clf()
        """
        metrics = ["efficiency", "sustainability", "dispersal", "following"]
        tasks = ["test_foraging", "test_exploration", "test_sustainability_low", "test_sustainability_high"]
        for task in tasks:
            fig, axs = plt.subplots(4,  figsize=(7, 12))

            for metric_idx, metric in enumerate(metrics):

                eval_results = {}
                for case_project in case_projects:
                    #last_gen = config[case_project]["num_gens"]
                    last_gen = 1400
                    ntrials = 3
                    total_frame = []
                    for trial in range(ntrials):
                        trial_project = case_project + "/trial_" + str(trial)
                        with open(trial_project + "/eval/data/gen_" + str(last_gen) + ".pkl", "rb") as f:
                            trial_results = pickle.load(f)
                            # make dataframe
                            metric_value = trial_results[metric][task]
                            gens = list(range(len(metric_value)))
                            trial = [trial] * len(metric_value)

                            trial_frame = pd.DataFrame({'metric': metric_value,
                                                        'gen': gens,
                                                        'trial': trial
                                                        })
                            total_frame.append(trial_frame)
                    total_frame = pd.concat(total_frame)
                    case_label = param_compare + "_" + str(configs[case_project][param_compare])

                    eval_results[case_label] = total_frame

                for case, result in eval_results.items():
                    sns.lineplot(data=result, x="gen", y="metric", label=case, ax=axs[metric_idx])
                axs[metric_idx].set_ylabel(metric)

            plt.savefig(case_dir + "/eval_test_" + task + ".png")

            plt.clf()
            plt.close()


def detect_tragedy(mean_rewards):
    window = 10
    tragedy = 0
    moving_average = np.convolve(mean_rewards, np.ones(window) / window, mode='valid')

    peak = np.max(moving_average)
    peak_loc = [idx for idx, el in enumerate(moving_average) if el == peak]
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
    for idx, el in enumerate(moving_average[int(local_minima_duration / 2):-local_minima_duration]):
        if el < 0.01:
            checksum = np.sum(
                [check for check in range(idx, idx + local_minima_duration) if moving_average[check] > 0.01])
            exists = np.sum(
                [1 for c in range(int(idx - local_minima_duration / 2), int(idx + local_minima_duration / 2)) if
                 (c in local_minima)])
            if not checksum and not exists:
                local_minima.append(idx)

    return tragedy, local_minima


def post_process(project_dir):
    post_process = {}

    try:
        with open(project_dir + "/trial_0/config.yaml", "r") as f:
            config = yaml.safe_load(f)

            last_gen = config["num_gens"] - 50
    except:
        print("no config for ", project_dir)
        return

    # last_gen = 500
    # load training data
    last_gen = 1400
    for trial in range(n_trials):
        trial_dir = project_dir + "/trial_" + str(trial)
        try:
            with open(trial_dir + "/train/data/gen_" + str(last_gen) + ".pkl", "rb") as f:
                results = pickle.load(f)
                post_process["tragedy_train"], post_process["local_minima"] = detect_tragedy(results["mean_rewards"])

            # load eval data
            with open(trial_dir + "/eval/data/gen_" + str(last_gen) + ".pkl", "rb") as f:
                results_eval = pickle.load(f)

                max_foraging = np.max(results_eval["efficiency"]["test_foraging"])
                last_foraging = results_eval["efficiency"]["test_foraging"][-1]

                epsilon = 2 * np.std(results_eval["efficiency"]["test_foraging"])
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

                    epsilon = 2 * np.std(results_eval["sustainability"][task])
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
                                                    "efficiency_last": efficiency_last}

                    with open(trial_dir + "/post_process.yaml", "w") as f:
                        yaml.dump(post_process, f)
                print("saving for project ", trial_dir)
        except:
            print("no data for project ", trial_dir)

    total_results = {}
    trial_results = []
    for trial in range(n_trials):
        trial_dir = project_dir + "/trial_" + str(trial)
        with open(trial_dir + "/post_process.yaml", "r") as f:
            trial_results.append(yaml.safe_load(f))

    total_results = {}
    low_efficiency_max = []
    low_efficiency_last = []
    high_efficiency_max = []
    high_efficiency_last = []

    for trial_result in trial_results:
        low_efficiency_max.append(trial_result["eval_test_sustainability_low"]["efficiency_max"])
        low_efficiency_last.append(trial_result["eval_test_sustainability_low"]["efficiency_last"])

        high_efficiency_max.append(trial_result["eval_test_sustainability_high"]["efficiency_max"])
        high_efficiency_last.append(trial_result["eval_test_sustainability_high"]["efficiency_last"])

    post_process = {"low_efficiency_max": {"mean": float(np.mean(low_efficiency_max)),
                                           "std": float(np.std(low_efficiency_max))},
                    "low_efficiency_last": {"mean": float(np.mean(low_efficiency_last)),
                                            "std": float(np.std(low_efficiency_last))},
                    "high_efficiency_max": {"mean": float(np.mean(high_efficiency_max)),
                                            "std": float(np.std(high_efficiency_max))},
                    "high_efficiency_last": {"mean": float(np.mean(high_efficiency_last)),
                                             "std": float(np.std(high_efficiency_last))}
                    }

    with open(project_dir + "/post_process.yaml", "w") as f:
        yaml.dump(post_process, f)


def compare(top_dir):
    all_results = {}
    project_dirs = [top_dir + "/" + el for el in os.listdir(top_dir) if os.path.isdir(top_dir + "/" + el)]
    project_dirs_cut = [el for el in os.listdir(top_dir) if os.path.isdir(top_dir + "/" + el)]
    for idx, project in enumerate(project_dirs):
        with open(project + '/post_process.yaml', "r") as f:
            results = yaml.safe_load(f)
            all_results[project_dirs_cut[idx]] = results

    # --- rank in sustainability tasks ---
    ranking_info = {}
    tests = ["eval_test_sustainability_low", "eval_test_sustainabiltiy_high"]
    for test in tests:
        max_effs = {}
        last_effs = {}
        for key, value in all_results.items():
            if test == "eval_test_sustainability_low":
                max_effs[key] = value["low_efficiency_max"]["mean"]
                last_effs[key] = value["low_efficiency_last"]["mean"]
            else:
                max_effs[key] = value["high_efficiency_max"]["mean"]
                last_effs[key] = value["high_efficiency_last"]["mean"]

        sorted_max_effs = np.sort(list(max_effs.values()))[::-1]
        sorted_projects_max = []
        for max_eff in sorted_max_effs:
            for key in max_effs.keys():
                if max_effs[key] == max_eff:
                    sorted_projects_max.append(key)
        sorted_last_effs = np.sort(list(last_effs.values()))[::-1]

        sorted_projects_last = []

        for last_eff in sorted_last_effs:
            for key in last_effs.keys():
                if last_effs[key] == last_eff:
                    sorted_projects_last.append(key)

        ranking_info[test] = {"sorted_last": sorted_projects_last,
                              "sorted_max": sorted_projects_max}
        print(ranking_info)
        # -------------------------------------------------------

    with open(top_dir + '/compare.yaml', "w") as f:
        yaml.dump(ranking_info, f)

def heatmap(top_dir, x_feature, y_feature):
    np.random.seed(0)
    sns.set()
    train_trials = 3
    current_gen = 1400
    project_dirs = [top_dir + "/" + el for el in os.listdir(top_dir) if os.path.isdir(top_dir + "/" + el)]
    data = {}
    metrics = ["efficiency"]
    test_types = ["test_foraging",
                  "test_exploration",
                  "test_following",
                  "test_sustainability_low",
                  "test_sustainability_high"]
    for test in test_types:
        for metric in metrics:
            for project in project_dirs:
                with open(project + "/trial_0/config.yaml", "r") as f:
                    config = yaml.safe_load(f)
                values = []
                for trial in range(train_trials):
                    with open(project + "/eval/data/gen_" + str(current_gen) + ".pkl", "wb") as f:
                        results = pickle.load(f)
                        values.append(results[test][metric][-1])
                data[config[x_feature], config[y_feature]] = np.mean(values)

            ser = pd.Series(list(data.values()),
                            index=pd.MultiIndex.from_tuples(data.keys()))
            df = ser.unstack().fillna(0)
            sns.heatmap(df)
            plt.show()
            save_dir = top_dir + "/test_" + test + "_metric_" + metric
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_dir + "/heatmap_" + x_feature + "_" + y_feature)



    # load all projects






if __name__ == "__main__":
    n_trials = 3
    top_dir = "projects/" + sys.argv[1]
    project_dirs = [top_dir + "/" + el for el in os.listdir(top_dir) if os.path.isdir(top_dir + "/" + el)]
    for project in project_dirs:
        post_process(project)

    # compare(top_dir)
    parameters = ["niches_scale", "regrowth_scale"]
    #for param in parameters:
        #compare_single_parameter(top_dir, param)
    #heatmap(top_dir, "regrowth_scale", "niches_scale")
