from abc import ABC, abstractmethod
from collections import defaultdict
import random
import sys
from typing import List, Dict, DefaultDict
from itertools import product
from copy import deepcopy
import numpy as np
from gym.spaces import Space, Box
from gym.spaces.utils import flatdim

from rl2021.exercise5.matrix_game import actions_to_onehot

def obs_to_tuple(obs):
    return tuple([tuple(o) for o in obs])


class MultiAgent(ABC):
    """Base class for multi-agent reinforcement learning
    """

    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        observation_spaces: List[Space],
        gamma: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of MARL agents
        namely epsilon, learning rate and discount rate.

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param observation_spaces (List[Space]): observation spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)

        :attr n_acts (List[int]): number of actions for each agent
        """

        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.observation_spaces = observation_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma

    @abstractmethod
    def act(self, obs: List[np.ndarray]) -> List[int]:
        """Chooses an action for all agents given observations

        :param obs (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :return (List[int]): index of selected action for each agent
        """
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def learn(self):
        ...


class IndependentQLearningAgents(MultiAgent):
    """Agent using the Independent Q-Learning algorithm
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of IndependentQLearningAgents

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents


        :attr q_tables (List[DefaultDict]): tables for Q-values mapping (OBS, ACT) pairs of observations
            and actions to respective Q-values for all agents

        Initializes some variables of the Independent Q-Learning agents, namely the epsilon, discount rate
        and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for i in range(self.num_agents)]


    def act(self, obss: List[np.ndarray]) -> List[int]:
        """Implement the epsilon-greedy action selection here

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :return (List[int]): index of selected action for each agent
        """
        # Storing actions of every agent
        actions = [0]*self.num_agents
        for agent in range(self.num_agents):
            # Finding best action for each agent
            a_vals = [self.q_tables[agent][(obss[agent], act)] for act in range(self.n_acts[agent])]
            max_val = max(a_vals)
            max_acts = [idx for idx, a_val in enumerate(a_vals) if a_val == max_val]

            # Exploration (e-greedy)
            if random.random() < self.epsilon:
                actions[agent] = random.randint(0, self.n_acts[agent] - 1)
            else: # Greedy update
                actions[agent] = random.choice(max_acts)
        return actions


    def learn(
        self, obss: List[np.ndarray], actions: List[int], rewards: List[float], n_obss: List[np.ndarray], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param n_obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the next environmental state for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current observation-action pair of each agent
        """
        # Storing updated q vals for each agent
        updated_values = [0]*self.num_agents
        for agent in range(self.num_agents):
            # Finding max q value for each possible action
            max_q_action = max([self.q_tables[agent][(n_obss[agent], a)] for a in range(self.n_acts[agent])])
            # Computing target
            target_value = rewards[agent] + self.gamma * (1 - dones[agent]) * max_q_action
            # Updating agent q table via Bellman equation
            self.q_tables[agent][(obss[agent], actions[agent])] += self.learning_rate * (
                target_value - self.q_tables[agent][(obss[agent], actions[agent])]
            )
            # Storing updated values
            updated_values[agent] = self.q_tables[agent][(obss[agent], actions[agent])]
        return updated_values


    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        pass


class JointActionLearning(MultiAgent):
    """Agents using the Joint Action Learning algorithm with Opponent Modelling
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of JointActionLearning

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr q_tables (List[DefaultDict]): tables for Q-values mapping (OBS, ACT) pairs of
            observations and joint actions to respective Q-values for all agents
        :attr models (List[DefaultDict[DefaultDict]]): each agent holding model of other agent
            mapping observation to other agent actions to count of other agent action

        Initializes some variables of the Joint Action Learning agents, namely the epsilon, discount
        rate and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_acts = [flatdim(action_space) for action_space in self.action_spaces]

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for _ in range(self.num_agents)]

        # initialise models for each agent mapping state to other agent actions to count of other agent action
        # in state
        self.models = [defaultdict(lambda: defaultdict(lambda: 0)) for _ in range(self.num_agents)] 

        # count observations - count for each agent
        self.c_obss = [defaultdict(lambda: 0) for _ in range(self.num_agents)]

    def get_ev(self, agent_idx: int, state: int, action: int):
        # Get joint actions of all other agents
        other_actions = deepcopy(self.n_acts) 
        del other_actions[agent_idx]
        # Enumerate all possible other agent actions 
        # For our case this will be [(0,), (1,), (2,)] since we have 1 other agent
        # and this agent can take 3 possible actions 
        other_action_combs = list(product(*[range(a) for a in other_actions]))
        ev = 0
        for other_comb in other_action_combs:
            # Grab one possible permutation of other agent actions
            pairs = list(deepcopy(other_comb))
            # Insert agent of interest's action into list
            pairs.insert(agent_idx, action)
            # Calculate EV components
            sa_pair_counts = self.models[agent_idx][state][other_comb]
            s_counts = self.c_obss[agent_idx][state]
            Q = self.q_tables[agent_idx][(state, tuple(pairs))] # Tuples are hashable, lists are not
            ev += Q * sa_pair_counts / s_counts if s_counts > 0 else 0 # Prevent 0 div errors
        return ev 

    def get_max_ev(self, agent_idx: int, state: int):
        """
            For a given agent, find the maximum expected 
            value considering all possible actions.
        """
        max_ev = float('-Inf') 
        for action in range(self.n_acts[agent_idx]):
            ev = self.get_ev(agent_idx, state, action)
            max_ev = max(max_ev, ev)
        return max_ev

    def act(self, obss: List[np.ndarray]) -> List[int]:
        """Implement the epsilon-greedy action selection here

        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :return (List[int]): index of selected action for each agent
        """
        joint_action = [0]*self.num_agents
        for agent in range(self.num_agents): 
            # Exploring (e-greedy update)
            if random.random() < self.epsilon:
                joint_action[agent] = random.randint(0, self.n_acts[agent]-1)
            # Greedy update
            else:
                evs = []
                # Get all expected values for each possible action
                for action in range(self.n_acts[agent]):
                    evs.append(self.get_ev(agent, obss[agent], action))
                # Find action associated with max expected value
                best_action = random.choice([idx for idx, ev in enumerate(evs) if ev == max(evs)])
                joint_action[agent] = best_action
        return joint_action

    def learn(
        self, obss: List[np.ndarray], actions: List[int], rewards: List[float], n_obss: List[np.ndarray], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables and models based on agents' experience


        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param n_obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the next environmental state for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current observation-action pair of each agent
        """
        updated_values = [0]*self.num_agents
        for agent in range(self.num_agents):
            # Update visitation counter
            self.c_obss[agent][obss[agent]] += 1
            # Update model (s)
            self.models[agent][obss[agent]][tuple(actions[:agent] + actions[agent+1:])] += 1

            # Update Q values via Bellman with modified ev targets
            max_ev = self.get_max_ev(agent, obss[agent])
            target_value = rewards[agent] + self.gamma * (1 - dones[agent]) * max_ev
            self.q_tables[agent][(obss[agent], tuple(actions))] += self.learning_rate * (
                target_value - self.q_tables[agent][(obss[agent], tuple(actions))]
            )
            updated_values[agent] = self.q_tables[agent][(obss[agent], tuple(actions))]
        return updated_values


    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        pass
