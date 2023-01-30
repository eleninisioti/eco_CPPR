# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
from abc import abstractmethod
from typing import Tuple
import jax.numpy as jnp



class TaskState(ABC):
    """A template of the task state."""
    obs: jnp.ndarray


class VectorizedTask(ABC):
    """Interface for all the EvoJAX tasks."""

    max_steps: int
    obs_shape: Tuple
    act_shape: Tuple
    test: bool
    multi_agent_training: bool = False

    @abstractmethod
    def reset(self, key: jnp.array) -> TaskState:
        """This resets the vectorized task.

        Args:
            key - A jax random key.
        Returns:
            TaskState. Initial task state.
        """
        raise NotImplementedError()

    @abstractmethod
    def step(self,
             state: TaskState,
             action: jnp.ndarray) -> Tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        """This steps once the simulation.

        Args:
            state - System internal states of shape (num_tasks, *).
            action - Vectorized actions of shape (num_tasks, action_size).
        Returns:
            TaskState. Task states.
            jnp.ndarray. Reward.
            jnp.ndarray. Task termination flag: 1 for done, 0 otherwise.
        """
        raise NotImplementedError()


from flax import linen as nn

import logging
import jax
import jax.numpy as jnp

import itertools
import functools

from typing import Tuple, Callable, List, Optional, Iterable, Any
from flax.struct import dataclass
from evojax.task.base import TaskState
from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.util import create_logger
from evojax.util import get_params_format_fn


class MetaRNN_bcppr(nn.Module):
    output_size: int
    out_fn: str
    hidden_layers: list
    encoder_in: bool
    encoder_layers: list

    def setup(self):

        self._num_micro_ticks = 1
        self._lstm = nn.recurrent.LSTMCell()
        self.convs = [nn.Conv(features=4, kernel_size=(3, 3), strides=2),
                      nn.Conv(features=8, kernel_size=(3, 3), strides=2)]

        self._hiddens = [(nn.Dense(size)) for size in self.hidden_layers]
        # self._encoder=nn.Dense(64)
        self._output_proj = nn.Dense(self.output_size)
        if (self.encoder_in):
            self._encoder = [(nn.Dense(size)) for size in self.encoder_layers]

    def __call__(self, h, c, inputs: jnp.ndarray, last_action: jnp.ndarray, reward: jnp.ndarray):
        carry = (h, c)
        # todo replace with scan
        # inputs=self._encoder(inputs)
        out = inputs
        for conv in self.convs:
            out = conv(out)
            out = nn.relu(out)
            out = nn.avg_pool(out, window_shape=(2, 2), strides=(1, 1))

        out = jnp.ravel(out)

        if (self.encoder_in):
            for layer in self._encoder:
                out = jax.nn.tanh(layer(out))

        inputs_encoded = jnp.concatenate([out, last_action, reward])

        for _ in range(self._num_micro_ticks):
            carry, out = self._lstm(carry, inputs_encoded)
        out = jnp.concatenate([inputs_encoded, out])
        for layer in self._hiddens:
            out = jax.nn.tanh(layer(out))
        out = self._output_proj(out)

        h, c = carry
        if self.out_fn == 'tanh':
            out = nn.tanh(out)
        elif self.out_fn == 'softmax':
            out = nn.softmax(out, axis=-1)
        else:
            if (self.out_fn != 'categorical'):
                raise ValueError(
                    'Unsupported output activation: {}'.format(self.out_fn))
        return h, c, out


@dataclass
class metaRNNPolicyState_bcppr(PolicyState):
    lstm_h: jnp.array
    lstm_c: jnp.array
    keys: jnp.array


class MetaRnnPolicy_bcppr(PolicyNetwork):

    def __init__(self, input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_act_fn: str = "categorical",
                 hidden_layers: list = [],
                 encoder: bool = False,
                 encoder_layers: list = [32, 32],
                 logger: logging.Logger = None):

        if logger is None:
            self._logger = create_logger(name='MetaRNNolicy')
        else:
            self._logger = logger
        model = MetaRNN_bcppr(output_dim, out_fn=output_act_fn, hidden_layers=hidden_layers, encoder_in=encoder,
                              encoder_layers=encoder_layers)
        self.params = model.init(jax.random.PRNGKey(0), jnp.zeros((hidden_dim)), jnp.zeros((hidden_dim)),
                                 jnp.zeros(input_dim), jnp.zeros([output_dim]), jnp.zeros([1]))

        self.num_params, format_params_fn = get_params_format_fn(self.params)
        self._logger.info('MetaRNNPolicy.num_params = {}'.format(self.num_params))
        self.hidden_dim = hidden_dim
        self._format_params_fn = jax.jit(jax.vmap(format_params_fn))
        self._forward_fn = jax.jit(jax.vmap(model.apply))

    def reset(self, states: TaskState) -> PolicyState:
        """Reset the policy.
        Args:
            TaskState - Initial observations.
        Returns:
            PolicyState. Policy internal states.
        """
        keys = jax.random.split(jax.random.PRNGKey(0), states.obs.shape[0])
        h = jnp.zeros((states.obs.shape[0], self.hidden_dim))
        c = jnp.zeros((states.obs.shape[0], self.hidden_dim))
        return metaRNNPolicyState_bcppr(keys=keys, lstm_h=h, lstm_c=c)

    def reset_b(self, obs: jnp.array) -> PolicyState:
        """Reset the policy.
        Args:
            TaskState - Initial observations.
        Returns:
            PolicyState. Policy internal states.
        """
        keys = jax.random.split(jax.random.PRNGKey(0), obs.shape[0])
        h = jnp.zeros((obs.shape[0], self.hidden_dim))
        c = jnp.zeros((obs.shape[0], self.hidden_dim))
        return metaRNNPolicyState_bcppr(keys=keys, lstm_h=h, lstm_c=c)

    def get_actions(self, t_states: TaskState, params: jnp.ndarray, p_states: PolicyState):
        params = self._format_params_fn(params)
        h, c, out = self._forward_fn(params, p_states.lstm_h, p_states.lstm_c, t_states.obs, t_states.last_actions,
                                     t_states.rewards)
        return out, metaRNNPolicyState_bcppr(keys=p_states.keys, lstm_h=h, lstm_c=c)


# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
from typing import Tuple
from PIL import Image
from PIL import ImageDraw
import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass

AGENT_VIEW = 7


@dataclass
class AgentStates(object):
    posx: jnp.uint16
    posy: jnp.uint16
    params: jnp.ndarray
    policy_states: PolicyState
    energy: jnp.ndarray
    time_good_level: jnp.uint16
    time_alive: jnp.uint16
    time_under_level: jnp.uint16
    alive: jnp.int8


@dataclass
class State(TaskState):
    obs: jnp.int8
    last_actions: jnp.int8
    rewards: jnp.int8
    state: jnp.int8
    agents: AgentStates
    steps: jnp.int32
    key: jnp.ndarray


def get_ob(state: jnp.ndarray, pos_x: jnp.int32, pos_y: jnp.int32) -> jnp.ndarray:
    obs = (jax.lax.dynamic_slice(jnp.pad(state, ((AGENT_VIEW, AGENT_VIEW), (AGENT_VIEW, AGENT_VIEW), (0, 0))),
                                 (pos_x - AGENT_VIEW + AGENT_VIEW, pos_y - AGENT_VIEW + AGENT_VIEW, 0),
                                 (2 * AGENT_VIEW + 1, 2 * AGENT_VIEW + 1, 3)))
    # obs=jnp.ravel(state)

    return obs


def get_init_state_fn(key: jnp.ndarray, SX, SY, posx, posy, pos_food_x, pos_food_y) -> jnp.ndarray:
    grid = jnp.zeros((SX, SY, 4))
    grid = grid.at[posx, posy, 0].add(1)
    grid = grid.at[posx[:5], posy[:5], 0].set(0)
    grid = grid.at[pos_food_x, pos_food_y, 1].set(1)
    grid = grid.at[:, :, 3].set(jnp.int8(jnp.clip(jnp.expand_dims((jnp.arange(0, SX)) / 30, 1), 0, 127)))

    # grid=grid.at[600:700,:300,3].set(0)
    # grid=grid.at[:,:,3].set(5)

    grid = grid.at[0, :, 2].set(1)
    grid = grid.at[-1, :, 2].set(1)
    grid = grid.at[:, 0, 2].set(1)
    grid = grid.at[:, -1, 2].set(1)
    return (grid)


get_obs_vector = jax.vmap(get_ob, in_axes=(None, 0, 0), out_axes=0)


class Gridworld(VectorizedTask):
    """gridworld task."""

    def __init__(self,
                 nb_agents: int = 100,
                 SX=300,
                 SY=100,
                 test: bool = False,
                 energy_decay=0.05,
                 max_age: int = 1000,
                 time_reproduce: int = 150,
                 time_death: int = 40,
                 max_ener=3.

                 ):
        self.obs_shape = (7, 7, 4)
        # self.obs_shape=11*5*4
        self.act_shape = tuple([5, ])
        self.test = test
        self.nb_agents = nb_agents
        self.SX = SX
        self.SY = SY
        self.energy_decay = energy_decay
        self.model = MetaRnnPolicy_bcppr(input_dim=((AGENT_VIEW * 2 + 1), (AGENT_VIEW * 2 + 1), 3), hidden_dim=4,
                                         output_dim=5, encoder_layers=[], hidden_layers=[8])

        self.energy_decay = energy_decay
        self.max_age = max_age
        self.time_reproduce = time_reproduce
        self.time_death = time_death
        self.max_ener = max_ener

        def reset_fn(key):
            next_key, key = random.split(key)
            posx = random.randint(next_key, (self.nb_agents,), 1, (SX - 1))
            next_key, key = random.split(key)
            posy = random.randint(next_key, (self.nb_agents,), 1, (SY - 1))
            next_key, key = random.split(key)

            pos_food_x = random.randint(next_key, (40 * self.nb_agents,), 1, (SX - 1))
            next_key, key = random.split(key)
            pos_food_y = random.randint(next_key, (40 * self.nb_agents,), 1, (SY - 1))
            next_key, key = random.split(key)
            grid = get_init_state_fn(key, SX, SY, posx, posy, pos_food_x, pos_food_y)

            next_key, key = random.split(key)

            params = jax.random.normal(
                next_key,
                (self.nb_agents, self.model.num_params,),
            ) / 100

            policy_states = self.model.reset_b(jnp.zeros(self.nb_agents, ))

            agents = AgentStates(posx=posx, posy=posy,
                                 energy=self.max_ener * jnp.ones((self.nb_agents,)).at[0:5].set(0),
                                 time_good_level=jnp.zeros((self.nb_agents,), dtype=jnp.uint16), params=params,
                                 policy_states=policy_states,
                                 time_alive=jnp.zeros((self.nb_agents,), dtype=jnp.uint16),
                                 time_under_level=jnp.zeros((self.nb_agents,), dtype=jnp.uint16),
                                 alive=jnp.ones((self.nb_agents,), dtype=jnp.uint16).at[0:self.nb_agents // 2].set(0))

            return State(state=grid, obs=get_obs_vector(grid, posx, posy), last_actions=jnp.zeros((self.nb_agents, 5)),
                         rewards=jnp.zeros((self.nb_agents, 1)), agents=agents,
                         steps=jnp.zeros((), dtype=int), key=next_key)

        self._reset_fn = jax.jit(reset_fn)

        def reproduce(params, posx, posy, energy, time_good_level, key, policy_states, time_alive, alive):
            # use agent 0 to 4 as a dump always dead if no dead put in there to be sure not overiding the alive ones
            # but maybe better to just make sure that there are 5 places available by checking if 5 dead (but this way may be better if we augment the 5)
            dead = 1 - alive
            dead = dead.at[0:5].set(0.001)

            next_key, key = random.split(key)
            # empty_spots for new agent are dead ones
            empty_spots = jax.random.choice(next_key, jnp.arange(time_good_level.shape[0]), p=dead, replace=False,
                                            shape=(5,))

            # compute reproducer spot
            next_key, key = random.split(key)
            reproducer = jnp.where(time_good_level > self.time_reproduce, 1, 0)
            reproducer = reproducer.at[0:5].set(0.001)
            reproducer_spots = jax.random.choice(next_key, jnp.arange(time_good_level.shape[0]),
                                                 p=reproducer / (reproducer.sum() + 1e-10), replace=False, shape=(5,))

            next_key, key = random.split(key)
            # new agents params with mutate , and also take pos of parents
            params = params.at[empty_spots].set(
                params[reproducer_spots] + 0.02 * jax.random.normal(next_key, (5, params.shape[1])))
            posx = posx.at[empty_spots].set(posx[reproducer_spots])
            posy = posy.at[empty_spots].set(posy[reproducer_spots])

            # new agents energy set at max

            # multiply by reproducer to be sure that the one that got selected by reproducer spot were reproducer indeed,
            # in case nb reproducer <5 but again maybe we can just check that at least 5 reproducer but weird
            energy = energy.at[empty_spots].set(self.max_ener * reproducer[reproducer_spots])
            energy = energy.at[0:5].set(0.)

            # new agents alive and time alive , time_good_alive, and RNN state set at 0

            alive = alive.at[empty_spots].set(1 * reproducer[reproducer_spots])
            time_alive = time_alive.at[empty_spots].set(0)
            time_good_level = time_good_level.at[empty_spots].set(0)
            policy_states = metaRNNPolicyState_bcppr(
                lstm_h=policy_states.lstm_h.at[empty_spots].set(jnp.zeros(policy_states.lstm_h.shape[1])),
                lstm_c=policy_states.lstm_c.at[empty_spots].set(jnp.zeros(policy_states.lstm_c.shape[1])),
                keys=policy_states.keys)

            # put time good level of reproducer back to 0
            # if in the dump don't put to 0 so that they can try reproduce in the next timestep
            time_good_level = time_good_level.at[reproducer_spots].set(
                time_good_level[reproducer_spots] * (empty_spots < 5))

            # kill the dump
            alive = alive.at[0:5].set(0)

            return (params, posx, posy, energy, time_good_level, policy_states, time_alive, alive)

        def step_fn(state):
            key = state.key
            next_key, key = random.split(key)

            # model selection of action
            actions_logit, policy_states = self.model.get_actions(state, state.agents.params,
                                                                  state.agents.policy_states)
            actions = jax.nn.one_hot(jax.random.categorical(next_key, actions_logit * 50, axis=-1), 5)

            grid = state.state
            energy = state.agents.energy
            alive = state.agents.alive

            # move agent
            action_int = actions.astype(jnp.int32)
            posx = state.agents.posx - action_int[:, 1] + action_int[:, 3]
            posy = state.agents.posy - action_int[:, 2] + action_int[:, 4]

            # wall
            hit_wall = state.state[posx, posy, 2] > 0
            posx = jnp.where(hit_wall, state.agents.posx, posx)
            posy = jnp.where(hit_wall, state.agents.posy, posy)

            posx = jnp.clip(posx, 0, SX - 1)
            posy = jnp.clip(posy, 0, SY - 1)
            grid = grid.at[state.agents.posx, state.agents.posy, 0].set(0)
            # add only the alive
            grid = grid.at[posx, posy, 0].add(1 * (alive > 0))

            ### collect food

            rewards = (alive > 0) * (grid[posx, posy, 1] > 0) * (1 / (grid[posx, posy, 0] + 1e-10))
            grid = grid.at[posx, posy, 1].add(-1 * (alive > 0))
            grid = grid.at[:, :, 1].set(jnp.clip(grid[:, :, 1], 0, 1))

            # regrow

            probability = jax.scipy.signal.convolve2d(grid[:, :, 1], jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4,
                                                      mode="same")
            overcrowded = jax.scipy.signal.convolve2d(grid[:, :, 1], np.ones((3, 3)), mode="same")
            probability = probability * (overcrowded < 5)
            # modulate the probability with the climate value
            probability = probability * jnp.clip(grid[:, :, 3] / 2000 - grid[:, :, 2], 0, 1)
            next_key, key = random.split(key)
            grid = grid.at[:, :, 1].add(random.bernoulli(next_key, probability))

            ####
            steps = state.steps + 1

            # decay of energy and clipping
            energy = energy - self.energy_decay + rewards
            energy = jnp.clip(energy, -1000, self.max_ener)

            time_good_level = jnp.where(energy > 0, (state.agents.time_good_level + 1) * alive, 0)
            time_under_level = jnp.where(energy < 0, state.agents.time_under_level + 1, 0)

            time_alive = state.agents.time_alive

            # look if still aliv
            alive = jnp.where(jnp.logical_or(time_alive > self.max_age, time_under_level > self.time_death), 0, alive)

            time_alive = time_alive + alive

            # compute reproducer and go through the function only if there is one
            reproducer = jnp.where(state.agents.time_good_level > self.time_reproduce, 1, 0)
            next_key, key = random.split(key)
            params, posx, posy, energy, time_good_level, policy_states, time_alive, alive = jax.lax.cond(
                reproducer.sum() > 0, reproduce, lambda y, z, a, b, c, d, e, f, g: (y, z, a, b, c, e, f, g), *(
                state.agents.params, posx, posy, energy, time_good_level, next_key, state.agents.policy_states,
                time_alive, alive))

            done = False
            steps = jnp.where(done, jnp.zeros((), jnp.int32), steps)
            cur_state = State(state=grid, obs=get_obs_vector(grid, posx, posy), last_actions=actions,
                              rewards=jnp.expand_dims(rewards, -1),
                              agents=AgentStates(posx=posx, posy=posy, energy=energy, time_good_level=time_good_level,
                                                 params=state.agents.params, policy_states=policy_states,
                                                 time_alive=time_alive, time_under_level=time_under_level, alive=alive),
                              steps=steps, key=key)
            # keep it in case we let agent several trials
            state = jax.lax.cond(
                done, lambda x: reset_fn(state.key), lambda x: x, cur_state)

            return state

        self._step_fn = jax.jit(step_fn)

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             ) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state)



