"""## env abstract class"""

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


"""# env but with walls on the border """

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
AGENT_VIEW = 5


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
    obs = jnp.ravel(jax.lax.dynamic_slice(jnp.pad(state, ((AGENT_VIEW, AGENT_VIEW), (AGENT_VIEW, AGENT_VIEW), (0, 0))),
                                          (pos_x - AGENT_VIEW + AGENT_VIEW, pos_y - AGENT_VIEW + AGENT_VIEW, 0),
                                          (2 * AGENT_VIEW + 1, 2 * AGENT_VIEW + 1, 3)))
    # obs=jnp.ravel(state)

    return obs


def get_init_state_fn(key: jnp.ndarray, SX, SY, posx, posy, pos_food_x, pos_food_y) -> jnp.ndarray:
    grid = jnp.zeros((SX, SY, 4))
    grid = grid.at[posx, posy, 0].add(1)
    grid = grid.at[pos_food_x, pos_food_y, 1].set(1)

    grid = grid.at[:, :, 3].set(1)

    grid = grid.at[0, :, 2].set(1)
    grid = grid.at[-1, :, 2].set(1)
    grid = grid.at[:, 0, 2].set(1)
    grid = grid.at[:, -1, 2].set(1)
    return (grid)


get_obs_vector = get_ob


class Gridworld(VectorizedTask):
    """gridworld task."""

    def __init__(self,
                 max_steps: int = 1000,
                 SX=300,
                 SY=100,
                 test: bool = False):
        self.max_steps = max_steps

        self.obs_shape = tuple([(AGENT_VIEW * 2 + 1) * (AGENT_VIEW * 2 + 1) * 3, ])
        # self.obs_shape=11*5*4
        self.act_shape = tuple([4, ])
        self.test = test
        self.SX = SX
        self.SY = SY

        def reset_fn(key):
            next_key, key = random.split(key)
            posx = random.randint(next_key, (1,), 1, (SX - 1) )
            next_key, key = random.split(key)
            posy = random.randint(next_key, (1,), 1, (SY - 1) )
            next_key, key = random.split(key)
            agents = AgentStates(posx=posx, posy=posy, seeds=jnp.zeros(1))

            pos_food_x = random.randint(next_key, (8,), 1, (SX - 1) )
            next_key, key = random.split(key)
            pos_food_y = random.randint(next_key, (8,), 1, (SY - 1) )

            next_key, key = random.split(key)
            grid = get_init_state_fn(key, SX, SY, posx, posy, pos_food_x, pos_food_y)

            return State(state=grid, obs=get_obs_vector(grid, posx[0], posy[0]), last_actions=jnp.zeros((4,)),
                         rewards=jnp.zeros((1,)), agents=agents,
                         steps=jnp.zeros((), dtype=int), key=next_key)

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, actions):
            grid = state.state

            # move agent

            action_int = actions.astype(jnp.int32)
            posx = state.agents.posx - action_int[0] + action_int[2]
            posy = state.agents.posy - action_int[1] + action_int[3]

            # wall
            hit_wall = state.state[posx, posy, 2] > 0
            posx = jnp.where(hit_wall, state.agents.posx, posx)
            posy = jnp.where(hit_wall, state.agents.posy, posy)

            posx = jnp.clip(posx, 0, SX - 1)
            posy = jnp.clip(posy, 0, SY - 1)
            grid = grid.at[state.agents.posx, state.agents.posy, 0].set(0)
            grid = grid.at[posx, posy, 0].add(1)

            ### collect food and seeds
            seeds = state.agents.seeds + jnp.int8((grid[posx, posy, 1] > 0))

            reward = (grid[posx[0], posy[0], 1] > 0) * (1 / (grid[posx[0], posy[0], 0] + 1e-10))
            grid = grid.at[posx, posy, 1].set(0)

            # regrow
            # num_neighbs = jax.scipy.signal.convolve2d(grid[:, :, 1], jnp.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
            #                                          mode="same")

            ####
            steps = state.steps + 1

            reset = (grid[:, :, 1].sum() < 1)
            done = steps > self.max_steps
            steps = jnp.where(done, jnp.zeros((), jnp.int32), steps)
            cur_state = State(state=grid, obs=get_obs_vector(grid, posx[0], posy[0]), last_actions=actions,
                              rewards=reward * jnp.ones((1,)),
                              agents=AgentStates(posx=posx, posy=posy, seeds=seeds),
                              steps=steps, key=key)
            # keep it in case we let agent several trials
            state = jax.lax.cond(
                reset, lambda x: reset_fn(state.key), lambda x: x, cur_state)

            return state, reward, done

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             action: jnp.ndarray) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)