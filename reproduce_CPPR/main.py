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
from reproduce_CPPR.agent import MetaRnnPolicy_b
import pickle

AGENT_VIEW = 3


def test(params, nb_agents, ind_best, SX, SY, key, model, project_dir, train_gen):
    SX=150
    SY=150
    params_b = params[ind_best[-nb_agents:]]
    rand_move = np.random.randint(4)
    moves = {0: "left", 1: "down", 2: "up", 3: "right"}
    print("random move in train_gen ", str(train_gen), str(moves[rand_move]))
    env = Gridworld(100, nb_agents, SX, SY)

    next_key, key = random.split(key)
    state = env.reset(next_key)

    policy_states = model.reset(state)
    eval_trials = 3
    test_dir = project_dir + "/evaluation/train_" + str(train_gen) + "_move_" + moves[rand_move]
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    hard_coded = int(nb_agents/10)

    for trial in range(eval_trials):
        print("trial ", str(trial))

        with VideoWriter( test_dir + "/trial_" + str(trial) + ".mp4", 4.0) as vid:

            for i in range(75):
                next_key, key = random.split(key)
                actions_logit, policy_states = model.get_actions(state, params_b, policy_states)
                if i < 20:
                    hard_agent = 0

                actions = jax.nn.one_hot(jax.random.categorical(next_key, actions_logit) , 5)

                # my agent
                total_hard_actions = jnp.zeros(shape=(hard_coded, 5))

                for hard_agent in range(hard_coded):
                    if i < 20:
                        hard_actions = jax.nn.one_hot([rand_move], 5)
                    else:
                        hard_actions = jax.nn.one_hot([rand_move], 5)

                    actions = actions.at[-hard_agent].set(hard_actions[0])


                cur_state, state, reward, done = env.step(state, actions)


                # print(state.agents.seeds)
                rgb_im = state.state[:, :, :3]

                rgb_im = np.repeat(rgb_im, 20, axis=0)
                rgb_im = np.repeat(rgb_im, 20, axis=1)
                vid.add(rgb_im)
            vid.close()




def stable_no_training_small():
    SX = int(640/4)
    SY = int(1520/4)
    nb_agents = 200
    num_train_gens = 500
    gen_length = 75
    env = Gridworld(num_train_gens*gen_length+1, nb_agents, SX, SY, climate_type="constant")
    key = jax.random.PRNGKey(np.random.randint(42))
    next_key, key = random.split(key)
    state = env.reset(next_key)

    plt.figure(figsize=(8, 6), dpi=160)

    vid = True
    project_dir = "projects/stable_no_training"
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    if (vid):
        with VideoWriter(project_dir + "/training.mp4", 20.0) as vid:

            for i in range(num_train_gens*gen_length):
                next_key, key = random.split(key)
                temp_actions = jax.nn.one_hot(random.randint(next_key, (nb_agents,), 0, 5), 5)
                actions = jax.nn.one_hot([3]*nb_agents, 5)

                cur_state, state, reward, done = env.step(state, actions)
                # print(state.agents.seeds)


                if i%10==0:
                    rgb_im = state.state[:, :, :3]
                    rgb_im = np.repeat(rgb_im, 5, axis=0)
                    rgb_im = np.repeat(rgb_im, 5, axis=1)
                    vid.add(rgb_im)
                    print("generation ", str(i))

def stable_training_old(fitness_criterion, selection_type):

    def rollout_base(params, key, state, iter=100):
        next_key, key = random.split(key)
        policy_states = model.reset(state)
        accumulated_rewards = jnp.zeros(params.shape[0])
        accumulated_staminas = jnp.zeros(params.shape[0])
        for i in range(iter):
            next_key, key = random.split(key)
            actions_logit, policy_states = model.get_actions(state, params, policy_states)
            actions = jax.nn.one_hot(jax.random.categorical(next_key, actions_logit, axis=-1), 5)
            cur_state, state, reward, done = env.step(state, actions)

            accumulated_rewards = accumulated_rewards + reward
            accumulated_staminas = accumulated_staminas*0.8 + reward
            accumulated_staminas = np.where(accumulated_staminas < 0.4, 0, accumulated_staminas)

        return accumulated_rewards, state, accumulated_staminas


    SX = int(640/4)
    SY = int(1520/4)
    nb_agents = 200
    num_train_gens = 700
    gen_length = 75
    env = Gridworld(num_train_gens*gen_length+1, nb_agents, SX, SY, climate_type="constant")
    key = jax.random.PRNGKey(np.random.randint(42))
    next_key, key = random.split(key)
    state = env.reset(next_key)
    #fitness_criterion = "rewards"

    plt.figure(figsize=(8, 6), dpi=160)

    vid = True
    project_dir = "projects/stable_training_small"
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    model = MetaRnnPolicy_b(input_dim=6 + ((AGENT_VIEW * 2 + 1) ** 2) * 3, hidden_dim=128, output_dim=4)

    params = jax.random.uniform(
        next_key,
        (nb_agents, model.num_params,),
        minval=-0.1,
        maxval=0.1,
    )

    mean_rewards = []
    mean_staminas = []
    gini_coeffs = []
    if (vid):
        try:
            vid = VideoWriter(project_dir + "/training_0.mp4", 20.0)

            for iter in range(num_train_gens):

                accumulated_rewards, state, accumulated_staminas = rollout_base(params, key, state, iter=gen_length)
                gini_coeffs.append(gini_coefficient(accumulated_rewards))

                mean_rewards.append(jnp.mean(accumulated_rewards))
                mean_staminas.append(jnp.mean(accumulated_staminas))
                if fitness_criterion == "stamina":
                    accumulated_rewards = accumulated_staminas

                ind_best = jnp.argsort(accumulated_rewards)

                if (iter % 10 == 0):
                    print("generation ", str(iter), str(mean_rewards[-1]))
                    print(jnp.mean(accumulated_rewards), accumulated_rewards[ind_best[-3:]],
                          accumulated_rewards[ind_best[:3]])

                next_key1, next_key2, next_key3, key = random.split(key, 4)
                params = params.at[ind_best[:3 * nb_agents // 4]].set(jnp.concatenate(
                    [params[ind_best[-nb_agents // 4:]] + 0.02 * jax.random.normal(next_key1,
                                                                                   (nb_agents // 4, params.shape[1])),
                     params[ind_best[-nb_agents // 4:]] + 0.02 * jax.random.normal(next_key2,
                                                                                   (nb_agents // 4, params.shape[1])),
                     params[ind_best[-nb_agents // 4:]] + 0.02 * jax.random.normal(next_key3,
                                                                                   (nb_agents // 4, params.shape[1]))]))

                rgb_im = state.state[:, :, :3]
                rgb_im = np.repeat(rgb_im, 5, axis=0)
                rgb_im = np.repeat(rgb_im, 5, axis=1)
                vid.add(rgb_im)

                if (iter % 50 == 0):
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

                    test(params, nb_agents, ind_best, SX, SY, key, model, project_dir, iter)


        except KeyboardInterrupt:
            print("running aborted")
            vid.close()

        with open(project_dir + "/data.pkl", "wb") as f:
            pickle.dump([mean_rewards, mean_staminas, gini_coeffs], file=f)

        with open(project_dir + "/for_eval.pkl", "wb") as f:
            pickle.dump([params, nb_agents, ind_best, SX, SY, key,  project_dir, iter], file=f)

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

        params = params.at[ind_best[:(-nb_agents //2)]].set(params[ind_best[-nb_agents // 2:]] +
                                                            0.02 * jax.random.normal(next_key1,
                                                                       (nb_agents // 2, params.shape[1])))

        params = params.at[ind_best[-nb_agents // 2:]].set(params[ind_best[-nb_agents // 2:]] +
                                                            0.02 * jax.random.normal(next_key1,
                                                                       (nb_agents // 2, params.shape[1])))

    return params

def stable_training(fitness_criterion, selection_type):

    def rollout_base(params, key, state, iter=100):
        next_key, key = random.split(key)
        policy_states = model.reset(state)
        accumulated_rewards = jnp.zeros(params.shape[0])
        accumulated_staminas = jnp.zeros(params.shape[0])
        for i in range(iter):
            next_key, key = random.split(key)
            actions_logit, policy_states = model.get_actions(state, params, policy_states)
            actions = jax.nn.one_hot(jax.random.categorical(next_key, actions_logit, axis=-1), 5)
            cur_state, state, reward, done = env.step(state, actions)

            accumulated_rewards = accumulated_rewards + reward
            accumulated_staminas = accumulated_staminas*0.8 + reward
            accumulated_staminas = np.where(accumulated_staminas < 0.4, 0, accumulated_staminas)

        return accumulated_rewards, state, accumulated_staminas


    SX = int(640)
    SY = int(1520)
    nb_agents = 200
    num_train_gens = 700
    gen_length = 75
    env = Gridworld(num_train_gens*gen_length+1, nb_agents, SX, SY, climate_type="constant")
    key = jax.random.PRNGKey(np.random.randint(42))
    next_key, key = random.split(key)
    state = env.reset(next_key)
    #fitness_criterion = "rewards"

    plt.figure(figsize=(8, 6), dpi=160)

    vid = True
    project_dir = "projects/stable_training_" + fitness_criterion + selection_type
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    model = MetaRnnPolicy_b(input_dim=6 + ((AGENT_VIEW * 2 + 1) ** 2) * 3, hidden_dim=128, output_dim=4)

    params = jax.random.uniform(
        next_key,
        (nb_agents, model.num_params,),
        minval=-0.1,
        maxval=0.1,
    )

    mean_rewards = []
    mean_staminas = []
    gini_coeffs = []
    if (vid):
        try:
            vid = VideoWriter(project_dir + "/training_0.mp4", 20.0)

            for iter in range(num_train_gens):

                accumulated_rewards, state, accumulated_staminas = rollout_base(params, key, state, iter=gen_length)
                gini_coeffs.append(gini_coefficient(accumulated_rewards))

                mean_rewards.append(jnp.mean(accumulated_rewards))
                mean_staminas.append(jnp.mean(accumulated_staminas))
                if fitness_criterion == "stamina":
                    accumulated_rewards = accumulated_staminas

                ind_best = jnp.argsort(accumulated_rewards)

                if (iter % 10 == 0):
                    print("generation ", str(iter), str(mean_rewards[-1]))
                    print(jnp.mean(accumulated_rewards), accumulated_rewards[ind_best[-3:]],
                          accumulated_rewards[ind_best[:3]])

                params = selection(params, nb_agents, key, ind_best, selection_type=selection_type)

                rgb_im = state.state[:, :, :3]
                rgb_im = np.repeat(rgb_im, 5, axis=0)
                rgb_im = np.repeat(rgb_im, 5, axis=1)
                vid.add(rgb_im)

                if (iter % 50 == 0):
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

                    test(params, nb_agents, ind_best, SX, SY, key, model, project_dir, iter)


        except KeyboardInterrupt:
            print("running aborted")
            vid.close()

        with open(project_dir + "/data.pkl", "wb") as f:
            pickle.dump([mean_rewards, mean_staminas, gini_coeffs], file=f)

        with open(project_dir + "/for_eval.pkl", "wb") as f:
            pickle.dump([params, nb_agents, ind_best, SX, SY, key,  project_dir, iter], file=f)

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
    #stable_no_training()
    #stable_no_training_small()
    rewards = "reward"
    selection_type = "gautier"
    print(selection_type, rewards)
    stable_training(rewards, selection_type)

    #stable_training("staminas", "eleni")
    #static_world(100)