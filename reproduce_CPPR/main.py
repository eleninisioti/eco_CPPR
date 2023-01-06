import os
import sys

sys.path.append(os.getcwd())
from reproduce_CPPR.utils import VideoWriter, gini_coefficient
from reproduce_CPPR.gridworld import Gridworld
import random
import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass
import matplotlib.pyplot as plt
import numpy as np
from reproduce_CPPR.agent import MetaRnnPolicy_b, DensePolicy_b
import pickle
import datetime

AGENT_VIEW = 3


def test(params, nb_agents, ind_best, SX, SY, key, model, project_dir, train_gen):
    SX = 150
    SY = 150
    nb_test_agents = 10
    params_b = params[ind_best[-nb_test_agents:]]
    rand_move = np.random.randint(4)
    rand_move = 3
    moves = {0: "left", 1: "down", 2: "up", 3: "right"}
    print("random move in train_gen ", str(train_gen), str(moves[rand_move]))
    init_food = 50
    env = Gridworld(100, nb_test_agents, init_food, SX, SY)

    next_key, key = random.split(key)
    state = env.reset(next_key)

    policy_states = model.reset(state)
    eval_trials = 3
    test_dir = project_dir + "/evaluation/train_" + str(train_gen) + "_move_" + moves[rand_move]
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    hard_coded = 5

    for trial in range(eval_trials):
        print("trial ", str(trial))

        with VideoWriter(test_dir + "/trial_" + str(trial) + ".mp4", 4.0) as vid:

            for i in range(75):
                next_key, key = random.split(key)
                actions_logit, policy_states = model.get_actions(state, params_b, policy_states)
                if i < 20:
                    hard_agent = 0

                actions = jax.nn.one_hot(jax.random.categorical(next_key, actions_logit), 5)

                # my agent

                for hard_agent in range(hard_coded):
                    if i < 20:
                        hard_actions = jax.nn.one_hot([rand_move], 5)
                    else:
                        hard_actions = jax.nn.one_hot([rand_move], 5)

                    actions = actions.at[hard_agent].set(hard_actions[0])

                cur_state, state, reward, done = env.step(state, actions)

                # print(state.agents.seeds)
                rgb_im = state.state[:, :, :3]

                rgb_im = np.repeat(rgb_im, 20, axis=0)
                rgb_im = np.repeat(rgb_im, 20, axis=1)
                vid.add(rgb_im)
            vid.close()


def stable_no_training_small():
    SX = int(640 / 4)
    SY = int(1520 / 4)
    nb_agents = 200
    num_train_gens = 1000
    gen_length = 75
    init_food = 100

    env = Gridworld(num_train_gens * gen_length + 1, nb_agents, init_food, SX, SY, climate_type="constant")
    key = jax.random.PRNGKey(np.random.randint(42))
    next_key, key = random.split(key)
    state = env.reset(next_key)

    plt.figure(figsize=(8, 6), dpi=160)

    vid = True
    project_dir = "projects/stable_no_training"
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    pos_x = state.agents.posx
    pos_y = state.agents.posy

    if (vid):
        with VideoWriter(project_dir + "/training.mp4", 20.0) as vid:
            for gen in range(num_train_gens):
                state = env._reset_fn_pos_food(next_key, pos_x, pos_y, state.state[:, :, 1])

                for i in range(gen_length):
                    next_key, key = random.split(key)
                    temp_actions = jax.nn.one_hot(random.randint(next_key, (nb_agents,), 0, 5), 5)
                    actions = jax.nn.one_hot([3] * nb_agents, 5)

                    cur_state, state, reward, done = env.step(state, actions)
                    # print(state.agents.seeds)

                rgb_im = state.state[:, :, :3]
                rgb_im = np.repeat(rgb_im, 5, axis=0)
                rgb_im = np.repeat(rgb_im, 5, axis=1)
                vid.add(rgb_im)
                print("generation ", str(gen))


def selection(params, nb_agents, key, ind_best, selection_type="Gautier"):
    if selection_type == "Gautier":

        next_key1, next_key2, next_key3, key = random.split(key, 4)
        params = params.at[ind_best[:3 * nb_agents // 4]].set(jnp.concatenate(
            [params[ind_best[-nb_agents // 4:]] + 0.02 * jax.random.normal(next_key1,
                                                                           (nb_agents // 4, params.shape[1])),
             params[ind_best[-nb_agents // 4:]] + 0.02 * jax.random.normal(next_key2,
                                                                           (nb_agents // 4, params.shape[1])),
             params[ind_best[-nb_agents // 4:]] + 0.02 * jax.random.normal(next_key3,
                                                                           (nb_agents // 4, params.shape[1]))]))

    else:
        next_key1, next_key2, next_key3, key = random.split(key, 4)

        params = params.at[ind_best[:(-nb_agents // 2)]].set(params[ind_best[-nb_agents // 2:]] +
                                                             0.02 * jax.random.normal(next_key1,
                                                                                      (
                                                                                          nb_agents // 2,
                                                                                          params.shape[1])))

        params = params.at[ind_best[-nb_agents // 2:]].set(params[ind_best[-nb_agents // 2:]] +
                                                           0.02 * jax.random.normal(next_key1,
                                                                                    (nb_agents // 2, params.shape[1])))

    return params


def rollout_base_b(env, model, params, pos_x, pos_y, state, key, iter=100):
    nb_agents = 200
    next_key, key = random.split(key)
    state = env._reset_fn_pos_food(next_key, pos_x, pos_y, state.state[:, :, 1])

    # temp_state = env._reset_fn_pos_food(next_key, pos_x, pos_y, food)
    policy_states = model.reset(state)
    accumulated_rewards = jnp.zeros(params.shape[0])
    accumulated_staminas = jnp.zeros(params.shape[0])

    for trial in range(iter):
        """
        next_key, key = random.split(key)
        actions_logit, policy_states = model.get_actions(state, params, policy_states)
        actions = jax.nn.one_hot(jax.random.categorical(next_key, actions_logit, axis=-1), 5)
        cur_state, state, reward, done = env.step(state, actions)
        accumulated_rewards = accumulated_rewards + reward
        accumulated_staminas = accumulated_staminas * 0.8 + reward
        accumulated_staminas = np.where(accumulated_staminas < 0.4, 0, accumulated_staminas)
        """

        next_key, key = random.split(key)
        actions = jax.nn.one_hot([3] * nb_agents, 5)

        cur_state, state, reward, done = env.step(state, actions)
        #

    return accumulated_rewards, state, accumulated_staminas


def stable_training(fitness_criterion, selection_type):
    SX = int(640 / 4)
    SY = int(1520 / 4)
    nb_agents = 200
    num_train_gens = 2000
    gen_length = 75
    init_food = 100
    policy = "dense"
    env = Gridworld(num_train_gens * gen_length + 1, nb_agents, init_food, SX, SY, climate_type="constant")
    key = jax.random.PRNGKey(np.random.randint(42))
    next_key, key = random.split(key)
    state = env.reset(next_key)
    # fitness_criterion = "rewards"

    plt.figure(figsize=(8, 6), dpi=160)

    vid = True
    now = datetime.datetime.now()
    today = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    project_dir = "projects/" + today + "/nofollow_" + fitness_criterion + selection_type + "_genlen_" + str(gen_length) + "_policy_" + policy
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    if policy == "metarnn":

        model = MetaRnnPolicy_b(input_dim=6 + ((AGENT_VIEW * 2 + 1) ** 2) * 3, hidden_dim=128, output_dim=4)
    else:
        model = DensePolicy_b(input_dim=6 + ((AGENT_VIEW * 2 + 1) ** 2) * 3, hidden_dim=128, output_dim=4)

    params = jax.random.uniform(
        next_key,
        (nb_agents, model.num_params,),
        minval=-0.1,
        maxval=0.1,
    )
    posx = state.agents.posx
    posy = state.agents.posy

    mean_rewards = []
    mean_staminas = []
    gini_coeffs = []
    if (vid):
        try:
            vid = VideoWriter(project_dir + "/training_0.mp4", 20.0)

            for iter in range(num_train_gens):

                next_key, key = random.split(key)
                state = env._reset_fn_pos_food(next_key, posx, posy, state.state[:, :, 1])

                # temp_state = env._reset_fn_pos_food(next_key, pos_x, pos_y, food)
                policy_states = model.reset(state)
                accumulated_rewards = jnp.zeros(params.shape[0])
                accumulated_staminas = jnp.ones(params.shape[0])

                for trial in range(gen_length):
                    next_key, key = random.split(key)
                    actions_logit, policy_states = model.get_actions(state, params, policy_states)
                    actions = jax.nn.one_hot(jax.random.categorical(next_key, actions_logit, axis=-1), 5)
                    none_actions = np.zeros((nb_agents, 5))
                    for agent_row, agent in enumerate(accumulated_staminas):
                        if agent == 0:
                            actions = actions.at[agent_row].set(np.zeros(5))
                    #actions = np.where(accumulated_staminas == 0, actions, actions)
                    cur_state, state, reward, done = env.step(state, actions)
                    accumulated_rewards = accumulated_rewards + reward
                    accumulated_staminas = accumulated_staminas * 0.8 + reward
                    accumulated_staminas = np.where(accumulated_staminas < 0.4, 0, accumulated_staminas)



                    #

                #accumulated_staminas = jnp.ones(nb_agents)
                #accumulated_rewards = jnp.ones(nb_agents)
                gini_coeffs.append(gini_coefficient(accumulated_rewards))

                mean_rewards.append(jnp.mean(accumulated_rewards))
                mean_staminas.append(jnp.mean(accumulated_staminas))
                if fitness_criterion == "stamina":
                    accumulated_rewards = accumulated_staminas

                ind_best = jnp.argsort(accumulated_rewards)

                if (iter % 10 == 0):
                    print("generation ", str(iter), str(mean_rewards[-1]))
                    print(jnp.mean(accumulated_rewards), accumulated_rewards[ind_best[-50:]],
                          accumulated_rewards[ind_best[:50]])

                params = selection(params, nb_agents, key, ind_best, selection_type=selection_type)
                posx = state.agents.posx
                posy = state.agents.posy
                """
                posx = jnp.concatenate(
                    [state.agents.posx[ind_best[-nb_agents // 4:]], state.agents.posx[ind_best[-nb_agents // 4:]],
                     state.agents.posx[ind_best[-nb_agents // 4:]], state.agents.posx[ind_best[-nb_agents // 4:]]])
                posy = jnp.concatenate(
                    [state.agents.posy[ind_best[-nb_agents // 4:]], state.agents.posy[ind_best[-nb_agents // 4:]],
                     state.agents.posy[ind_best[-nb_agents // 4:]], state.agents.posy[ind_best[-nb_agents // 4:]]])
                """
                rgb_im = state.state[:, :, :3]
                rgb_im = np.repeat(rgb_im, 5, axis=0)
                rgb_im = np.repeat(rgb_im, 5, axis=1)
                vid.add(rgb_im)

                if (iter % 10 == 0):
                    vid.close()
                    vid = VideoWriter(project_dir + "/training_" + str(iter) + ".mp4", 20.0)
                    with open(project_dir + "/data_" + str(iter) + ".pkl", "wb") as f:
                        pickle.dump([mean_rewards, mean_staminas, gini_coeffs], file=f)

                    plt.plot(range(len(mean_rewards)), mean_rewards)
                    plt.ylabel("reward")
                    plt.savefig(project_dir + "/rewards_" + str(iter) + ".png")
                    plt.clf()

                    plt.plot(range(len(mean_staminas)), mean_staminas)
                    plt.ylabel("reward")
                    plt.savefig(project_dir + "/staminas_" + str(iter) + ".png")
                    plt.clf()

                    plt.plot(range(len(gini_coeffs)), gini_coeffs)
                    plt.ylabel("Equalilty")
                    plt.savefig(project_dir + "/equality_" + str(iter) + ".png")
                    plt.clf()

                if (iter % 50 == 0):

                    test(params, nb_agents, ind_best, SX, SY, key, model, project_dir, iter)


        except KeyboardInterrupt:
            print("running aborted")
            vid.close()

        with open(project_dir + "/data.pkl", "wb") as f:
            pickle.dump([mean_rewards, mean_staminas, gini_coeffs], file=f)

        with open(project_dir + "/for_eval.pkl", "wb") as f:
            pickle.dump([params, nb_agents, ind_best, SX, SY, key, project_dir, iter], file=f)

        test(params, nb_agents, ind_best, SX, SY, key, model, project_dir, iter)

        plt.plot(range(len(mean_rewards)), mean_rewards)
        plt.ylabel("reward")
        plt.savefig(project_dir + "/rewards.png")
        plt.clf()

        plt.plot(range(len(mean_staminas)), mean_staminas)
        plt.ylabel("reward")
        plt.savefig(project_dir + "/staminas.png")
        plt.clf()

        plt.plot(range(len(gini_coeffs)), gini_coeffs)
        plt.ylabel("Equalilty")
        plt.savefig(project_dir + "/equality.png")
        plt.clf()


if __name__ == "__main__":
    # stable_no_training()
    #stable_no_training_small()
    rewards = "reward"
    selection_type = "gautier"
    # rewards = "stamina"
    # selection_type = "gautier"
    print(selection_type, rewards)
    stable_training(rewards, selection_type)

    # stable_training("staminas", "eleni")
    # static_world(100)
