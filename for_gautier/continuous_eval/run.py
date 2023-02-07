import sys
import datetime
import os
import sys

sys.path.append(os.getcwd())
import yaml
from continuous_eval.train import train
from continuous_eval.eval import eval_pretrained

from continuous_eval.utils import create_jzscript
import copy
import pickle


def setup_project(config, exp_name):
    now = datetime.datetime.now()
    today = str(now.day) + "_" + str(now.month) + "_" + str(now.year)

    if mode == "local":
        top_dir = "projects/"
    else:
        top_dir = "/gpfsscratch/rech/imi/" + user + "/CPPR_log/projects/"
    project_dir = top_dir + today + "/" + exp_name
    conf_test = copy.copy(config)



    project_dir += "/"

    for key, value in config.items():
        if key != "trial" and key != "load_trained":
            project_dir += key + "_" + str(value)

    project_dir += "/trial_" + str(config["trial"])

    if not os.path.exists(project_dir + "/train/data"):
        os.makedirs(project_dir + "/train/data")

    if not os.path.exists(project_dir + "/train/models"):
        os.makedirs(project_dir + "/train/models")

    if not os.path.exists(project_dir + "/train/media"):
        os.makedirs(project_dir + "/train/media")

    if not os.path.exists(project_dir + "/eval/data"):
        os.makedirs(project_dir + "/eval/data")

    if not os.path.exists(project_dir + "/eval/media"):
        os.makedirs(project_dir + "/eval/media")

    print("Saving current simulation under ", project_dir)

    with open(project_dir + "/config.yaml", "w") as f:
        yaml.dump(config, f)
    return project_dir

def process_paper():
    ##with open(project_dir + "/config.yaml", "r") as f:
    #    config = pickle.safe_load(f)
    config["trial"] = 0
    config["agent_view"] = 7
    config["num_gens"] = 980
    config["eval_freq"] = 20
    config["nb_agents"] = 1000

    project_dir = setup_project(config, "process_paper")

    if mode == "local":
        eval_pretrained(project_dir)

    elif mode == "server":
        create_jzscript(project_dir, user, mode="eval")


if __name__ == "__main__":
    mode = sys.argv[1]  # choose between local and server
    user = sys.argv[2]

    config = {"nb_agents": 0,
              "num_gens": 2000,
              "eval_freq": 50,
              "gen_length": 0,
              "grid_width": 0,
              "init_food": 0,
              "agent_view": 3,
              "regrowth_scale": 0,
              "niches_scale": 2}

    # test()
    # limited_parametric()
    process_paper()
