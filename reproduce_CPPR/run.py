import sys
import datetime
import os
import sys
sys.path.append(os.getcwd())
import yaml
from reproduce_CPPR.train import train
from reproduce_CPPR.utils import create_jzscript

def setup_project(config, exp_name):
    now = datetime.datetime.now()
    today = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    if mode == "local":
        top_dir = "projects/"
    else:
        top_dir = "/gpfsscratch/rech/imi/" + user + "/CPPR_log/projects/"
    project_dir = top_dir + today + "/" + exp_name + "/"
    for key, value in config.items():
        project_dir += key + "_" + str(value)




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


def test():
    config["num_gens"] = 1000
    config["nb_agents"] = 200
    config["grid_length"] = 320
    config["grid_width"] = 160
    config["gen_length"] = 750
    config["init_food"] = 250
    config["regrowth_scale"] = 0.0005
    project_dir = setup_project(config, "test")
    if mode == "local":
        train(project_dir)

    elif mode == "server":
        create_jzscript(project_dir, user)


def parametric():
    config["num_gens"] = 2000
    gen_length_values = [500, 1000]
    nb_agents_values = [20, 200, 600]
    world_size_values = [{"width": 160, "length": 380, "init_food": 500},
                         {"width": int(160*2/3), "length": int(380*2/3), "init_food": int(500*2/3)}]
    niches_scale_values = [2, 20, 200]
    regrowth_scale_values = [0.002, 0.0005]
    n_trials = 2

    for trial in range(n_trials):

        for gen_length in gen_length_values:

            for nb_agent in nb_agents_values:
                for world_size in world_size_values:
                    for niches_scale in niches_scale_values:
                        for regrowth_scale in regrowth_scale_values:
                            config["gen_length"] = gen_length
                            config["nb_agents"] = nb_agent
                            config["grid_length"] = world_size["length"]
                            config["grid_width"] = world_size["width"]
                            config["init_food"] = world_size["init_food"]
                            config["niches_scale"] = niches_scale
                            config["regrowth_scale"] = regrowth_scale
                            config["trial"] = trial

                            project_dir = setup_project(config, "parametric")

                            if mode == "local":
                                train(project_dir)

                            elif mode == "server":
                                create_jzscript(project_dir, user)


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

    #test()
    parametric()