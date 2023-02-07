import os
import sys

sys.path.append(os.getcwd())

from reproduce_CPPR.agent import MetaRnnPolicy_bcppr
from reproduce_CPPR.gridworld import Gridworld, ACTION_SIZE

from reproduce_CPPR.utils import VideoWriter
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime
from reproduce_CPPR.testing import eval, process_eval
from evojax.util import save_model, load_model
import yaml
import time



def eval_pretrained(project_dir):
    with open(project_dir + "/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    key = jax.random.PRNGKey(np.random.randint(42))

    gens = list(range(0,config["num_gens"], config["eval_freq"]))

    eval_params = []

    model = MetaRnnPolicy_bcppr(input_dim=((config["agent_view"] * 2 + 1), (config["agent_view"] * 2 + 1), 3),
                                hidden_dim=4,
                                output_dim=ACTION_SIZE,
                                encoder_layers=[],
                                hidden_layers=[8])

    for gen in gens:
        next_key, key = random.split(key)


        params, obs_param = load_model(project_dir + "/train/models", "gen_" + str(gen) + ".npz")

        # run offline evaluation
        eval_params.append(eval(params, config["nb_agents"], key, model, project_dir, config["agent_view"], gen))
        process_eval(eval_params, project_dir, gen)


if __name__ == "__main__":
    project_dir = sys.argv[1]

    eval(project_dir)
