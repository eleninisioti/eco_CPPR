
from abc import ABC
from abc import abstractmethod
from typing import Tuple
import jax.numpy as jnp
import math

class TaskState(ABC):
    """A
Best,template of the task state."""
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

SIZE_GRID = 4
AGENT_VIEW = 3


@dataclass
class AgentStates(object):
    posx: jnp.int8
    posy: jnp.int8
    seeds: jnp.int8


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


def get_init_state_fn(key: jnp.ndarray, SX, SY, posx, posy, pos_food_x, pos_food_y, climate_type, climate_var,
                      scale_niches, scale_niches_exponential, gen=0) -> jnp.ndarray:
    grid = jnp.zeros((SX, SY, 4))
    grid = grid.at[posx, posy, 0].add(1)
    grid = grid.at[pos_food_x, pos_food_y, 1].set(1)
    # change climate
    if climate_type == "noisy":
        new_array = jnp.clip(np.arange(0, SX) / SX, 0, 1)
        for col in range(SY - 1):
            new_col = jnp.clip(np.arange(0, SX) / SX, 0, 1)
            new_array = jnp.append(new_array, new_col)
        new_array = jnp.transpose(jnp.reshape(new_array, (SY, SX)))
        grid = grid.at[:, :, 3].set(new_array)

        baseline = jax.random.normal(key, (grid.shape[0], grid.shape[1])) * climate_var
        # grid = grid.at[:, :, 3].add(baseline)

    elif climate_type == "constant":
        if scale_niches_exponential > 1:
            alpha=scale_niches_exponential
            new_array = jnp.clip(np.asarray([(math.pow(alpha, el) - 1) / (alpha - 1) for el in np.arange(0, SX) / SX]), 0,
                               1)
        else:
            new_array= jnp.clip(np.arange(0, SX) / SX * scale_niches, 0, 1)

        for col in range(SY - 1):
            if scale_niches_exponential > 1:
                alpha = scale_niches_exponential
                new_col = jnp.clip(np.asarray([(math.pow(alpha, el)-1)/(alpha-1)  for el in np.arange(0, SX) / SX]), 0, 1)
                #new_col = jnp.where(new_col < 0.2, 0, new_col)
            else:
                new_col = jnp.clip(np.arange(0, SX) / SX*scale_niches, 0, 1)

            new_array = jnp.append(new_array, new_col)
        new_array = jnp.transpose(jnp.reshape(new_array, (SY, SX)))
        grid = grid.at[:, :, 3].set(new_array)
        #grid = grid.at[:, :, 3].set(0)

    elif climate_type == "periodic":
        period = 2000
        omega = 2 * np.pi / period
        amplitude = 2
        new_array = jnp.clip(np.arange(0, SX) / SX, 0, 1)
        for col in range(SY - 1):
            new_col = jnp.clip(np.arange(0, SX) / SX, 0, 1)
            new_array = jnp.append(new_array, new_col)
        new_array = jnp.transpose(jnp.reshape(new_array, (SY, SX)))
        scale = amplitude * jnp.sin(gen * omega)
        print("scale", scale)
        # scale=-1
        grid = grid.at[:, :, 3].set(new_array + scale)

    elif climate_type == "no-niches" or climate_type == "no-regrowth":
        grid = grid.at[:, :, 3].set(1.0)

    # grid=grid.at[:,:,3].set(5)

    grid = grid.at[0, :, 2].set(1)
    grid = grid.at[-1, :, 2].set(1)
    grid = grid.at[:, 0, 2].set(1)
    grid = grid.at[:, -1, 2].set(1)
    return (grid)


get_obs_vector = jax.vmap(get_ob, in_axes=(None, 0, 0), out_axes=0)


class GridworldDynamic(VectorizedTask):
    """gridworld task."""

    def __init__(self,
                 max_steps: int = 1000,
                 nb_agents: int = 100,
                 SX=300,
                 SY=100,
                 init_food = 200,
                 climate_type="no-niches",
                 climate_var=0.2,
                 place_agent=False,
                 place_resources=False,
                 scale_niches = 1,
                 scale_niches_exponential=0,
                 regrowth_scale = 0.0005,
                 test: bool = False):
        self.max_steps = max_steps

        self.obs_shape = (7, 7, 4)
        # self.obs_shape=11*5*4
        self.act_shape = tuple([4, ])
        self.test = test
        self.nb_agents = nb_agents
        self.SX = SX
        self.SY = SY
        self.climate_type=climate_type
        self.climate_var = climate_var
        self.regrowth_scale = regrowth_scale
        self.place_agent = place_agent
        self.place_resources = place_resources

        def reset_fn(key):
            if self.place_agent:
                next_key, key = random.split(key)
                posx = random.randint(next_key, (nb_agents,), int(2/5*SX), int(3/5*SX))
                next_key, key = random.split(key)
                posy = random.randint(next_key, (nb_agents,), int(2/5*SX), int(3/5*SX))
                next_key, key = random.split(key)
                agents = AgentStates(posx=posx, posy=posy, seeds=jnp.zeros(nb_agents))
            else:
                next_key, key = random.split(key)
                posx = random.randint(next_key, (nb_agents,), 1, (SX - 1))
                next_key, key = random.split(key)
                posy = random.randint(next_key, (nb_agents,), 1, (SY - 1))
                next_key, key = random.split(key)
                agents = AgentStates(posx=posx, posy=posy, seeds=jnp.zeros(nb_agents))


            if self.place_resources:

                pos_food_x = jnp.concatenate((random.randint(next_key, (int(init_food/4),), 4*(int(SX/5)), (SX - 1)),
                                             random.randint(next_key, (int(init_food/4),), 1, 1*(int(SX/5))),
                                              random.randint(next_key, (int(init_food/4),), 1, (SX - 1)),
                                              random.randint(next_key, (int(init_food/4),), 1, (SX - 1))))
                next_key, key = random.split(key)
                pos_food_y = jnp.concatenate((random.randint(next_key, (int(init_food/4),), 1, SY-1),
                                              random.randint(next_key, (int(init_food/4),), 1, SY-1),
                                              random.randint(next_key, (int(init_food/4),), 4*(int(SY/5)), (SY - 1)),
                                             random.randint(next_key, (int(init_food/4),), 1, 1*(int(SY/5)))))
                next_key, key = random.split(key)

            else:
                pos_food_x = random.randint(next_key, (init_food,), 1, (SX - 1))
                next_key, key = random.split(key)
                pos_food_y = random.randint(next_key, (init_food,), 1, (SY - 1))
                next_key, key = random.split(key)


            grid = get_init_state_fn(key, SX, SY, posx, posy, pos_food_x, pos_food_y, climate_type,
                                     climate_var, scale_niches, scale_niches_exponential)

            return State(state=grid, obs=get_obs_vector(grid, posx, posy), last_actions=jnp.zeros((nb_agents, 4)),
                         rewards=jnp.zeros((nb_agents, 1)), agents=agents,
                         steps=jnp.zeros((), dtype=int), key=next_key)

        self._reset_fn = jax.jit(reset_fn)

        def reset_fn_pos_food(key, posx, posy, food, gen):


            next_key, key = random.split(key)
            agents = AgentStates(posx=posx, posy=posy, seeds=jnp.zeros(nb_agents))
            """
            next_key, key = random.split(key)
            posx = random.randint(next_key, (nb_agents,), 1, (SX - 1))
            next_key, key = random.split(key)
            posy = random.randint(next_key, (nb_agents,), 1, (SY - 1))
            next_key, key = random.split(key)
            agents = AgentStates(posx=posx, posy=posy, seeds=jnp.zeros(nb_agents))
            

            pos_food_x = random.randint(next_key, (init_food,), 1, SX - 1)
            next_key, key = random.split(key)
            pos_food_y = random.randint(next_key, (init_food,), 1, SY - 1)
            next_key, key = random.split(key)
            grid = get_init_state_fn(key, SX, SY, posx, posy, pos_food_x, pos_food_y, self.climate_type,
                                     self.climate_var, gen)
            grid = grid.at[:, :, 1].set(food)
            """
            pos_food_x = random.randint(next_key, (init_food,), 1, (SX - 1))
            next_key, key = random.split(key)
            pos_food_y = random.randint(next_key, (init_food,), 1, (SY - 1))
            next_key, key = random.split(key)
            grid = get_init_state_fn(key, SX, SY, posx, posy, pos_food_x, pos_food_y, climate_type, climate_var, scale_niches, scale_niches_exponential, gen)
            return State(state=grid, obs=get_obs_vector(grid, posx, posy), last_actions=jnp.zeros((nb_agents, 4)),
                         rewards=jnp.zeros((nb_agents, 1)), agents=agents,
                         steps=jnp.zeros((), dtype=int), key=next_key)

        self._reset_fn_pos_food = jax.jit(reset_fn_pos_food)

        def step_fn(state, actions):
            grid = state.state

            # move agent
            # maybe later make the agent to output the one hot categorical
            action_int = actions.astype(jnp.int32)
            posx = state.agents.posx - action_int[:, 0] + action_int[:, 2]
            posy = state.agents.posy - action_int[:, 1] + action_int[:, 3]

            # wall
            hit_wall = state.state[posx, posy, 2] > 0
            posx = jnp.where(hit_wall, state.agents.posx, posx)
            posy = jnp.where(hit_wall, state.agents.posy, posy)

            posx = jnp.clip(posx, 0, SX - 1)
            posy = jnp.clip(posy, 0, SY - 1)
            grid = grid.at[state.agents.posx, state.agents.posy, 0].set(0)
            grid = grid.at[posx, posy, 0].set(1)

            ### collect food and seeds
            seeds = state.agents.seeds + jnp.int8((grid[posx, posy, 1] > 0))

            rewards = (grid[posx, posy, 1] > 0) * (1 / (grid[posx, posy, 0] + 1e-10))
            grid = grid.at[posx, posy, 1].set(0)

            # regrow
            num_neighbs = jax.scipy.signal.convolve2d(grid[:, :, 1], jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
                                                      mode="same")
            scale = grid[:, :, 3]
            scale_constant = regrowth_scale
            num_neighbs = jnp.where(num_neighbs == 0, 0, num_neighbs)
            num_neighbs = jnp.where(num_neighbs == 1, 0.01 / 5, num_neighbs)
            num_neighbs = jnp.where(num_neighbs == 2, 0.01 / scale_constant, num_neighbs)
            num_neighbs = jnp.where(num_neighbs == 3, 0.05 / scale_constant, num_neighbs)
            # num_neighbs = jnp.where(num_neighbs == 4, 0.05/scale_constant, num_neighbs)
            num_neighbs = jnp.where(num_neighbs > 3, 0, num_neighbs)
            # print(jnp.sum(num_neighbs))
            num_neighbs = jnp.multiply(num_neighbs, scale)
            num_neighbs = jnp.where(num_neighbs > 0, num_neighbs, 0)
            next_key, key = random.split(state.key)

            if climate_type != "no-regrowth":
                grid = grid.at[:, :, 1].add(random.bernoulli(next_key, num_neighbs))

            # cells with too many resources around them die
            num_neighbs_subtract = jax.scipy.signal.convolve2d(grid[:, :, 1],
                                                               jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
                                                               mode="same")
            scale = grid[:, :, 3]
            scale_constant = 1
            num_neighbs_subtract = jnp.where(num_neighbs_subtract > 3, 0.01 / scale_constant, num_neighbs_subtract)
            num_neighbs_subtract = jnp.where(num_neighbs_subtract <= 3, 0, num_neighbs_subtract)
            num_neighbs_subtract = jnp.multiply(num_neighbs_subtract, scale)

            if climate_type != "no-regrowth":

                grid = grid.at[:, :, 1].add(-1 * random.bernoulli(next_key, num_neighbs_subtract))

            # resources die after some time
            # discount = 0.0001
            discount = 0.0
            alive_cells = jnp.where(grid[:, :, 1] > 0, discount, 0)

            if climate_type != "no-regrowth":

                grid = grid.at[:, :, 1].add(-1 * random.bernoulli(next_key, alive_cells))

            # print("after", jnp.sum(num_neighbs))
            # modulate the probability with the climate value
            # probability=probability*jnp.clip(grid[:,:,3]/2000-grid[:,:,2],0,1)
            # grid=grid.at[:,:,1].add(random.bernoulli(next_key, num_neighbs))
            # grid = grid.at[:, :, 1].add(random.bernoulli(next_key, num_neighbs_subtract))

            ####
            steps = state.steps + 1

            done = (steps > self.max_steps)
            steps = jnp.where(done, jnp.zeros((), jnp.int32), steps)
            cur_state = State(state=grid, obs=get_obs_vector(grid, posx, posy), last_actions=actions,
                              rewards=jnp.expand_dims(rewards, -1),
                              agents=AgentStates(posx=posx, posy=posy, seeds=seeds),
                              steps=steps, key=key)
            # keep it in case we let agent several trials
            state = jax.lax.cond(
                done, lambda x: reset_fn(state.key), lambda x: x, cur_state)

            return cur_state, state, rewards, done

        self._step_fn = jax.jit(step_fn)

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             action: jnp.ndarray) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)