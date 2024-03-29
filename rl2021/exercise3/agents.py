from abc import ABC, abstractmethod
from copy import deepcopy
import gym
import numpy as np
import os.path
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, Iterable, List
from torch.autograd import Variable
from rl2021.exercise3.networks import FCNetwork
from rl2021.exercise3.replay import Transition, ReplayBuffer
import random


class Agent(ABC):
    """Base class for Deep RL Exercise 3 Agents

    :attr action_space (gym.Space): action space of used environment
    :attr observation_space (gym.Space): observation space of used environment
    :attr saveables (Dict[str, torch.nn.Module]):
        mapping from network names to PyTorch network modules

    Note:
        see http://gym.openai.com/docs/#spaces for more information on Gym spaces
    """

    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        """The constructor of the Agent Class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        """
        self.action_space = action_space
        self.observation_space = observation_space

        self.saveables = {}

    def save(self, path: str, suffix: str = "") -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models_{suffix}.pt"
        where suffix is given by the optional parameter (by default empty string "")

        :param path (str): path to directory where to save models
        :param suffix (str, optional): suffix given to models file
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path

    def restore(self, save_path: str):
        """Restores PyTorch models from models file given by path

        :param save_path (str): path to file containing saved models
        """
        dirname, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dirname, save_path)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    @abstractmethod
    def act(self, obs: np.ndarray):
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
    def update(self):
        ...


class DQN(Agent):
    """The DQN agent for exercise 3
    :attr critics_net (FCNetwork): fully connected DQN to compute Q-value estimates
    :attr critics_target (FCNetwork): fully connected DQN target network
    :attr critics_optim (torch.optim): PyTorch optimiser for DQN critics_net
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr update_counter (int): counter of updates for target network updates
    :attr target_update_freq (int): update frequency (number of iterations after which the target
        networks should be updated)
    :attr batch_size (int): size of sampled batches of experience
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        target_update_freq: int,
        batch_size: int,
        gamma: float,
        use_lunar_scheduler: bool,
        **kwargs,
    ):
        """The constructor of the DQN agent class
        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param target_update_freq (int): update frequency (number of iterations after which the target
            networks should be updated)
        :param batch_size (int): size of sampled batches of experience
        :param gamma (float): discount rate gamma
        """
        super().__init__(action_space, observation_space)

        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.critics_net = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=None
        )

        self.critics_target = deepcopy(self.critics_net)

        self.critics_optim = Adam(
            self.critics_net.parameters(), lr=learning_rate, eps=1e-3
        )

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.update_counter = 0
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma
        self.use_lunar_scheduler = use_lunar_scheduler

        # ######################################### #

        self.saveables.update(
            {
                "critics_net": self.critics_net,
                "critics_target": self.critics_target,
                "critic_optim": self.critics_optim,
            }
        )

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        if self.use_lunar_scheduler:
            self.epsilon = 1.0
            if timestep > 0.10*max_timestep:
                self.epsilon = 1.0 - (min(1.0, timestep/(0.25 * max_timestep))) * 0.99995
        else:
            # Linear decay scheduler
            self.epsilon = 1.0 - (min(1.0, timestep/(0.3 * max_timestep))) * 0.97

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        When explore is False you should select the best action possible (greedy). However, during
        exploration, you should be implementing an exploration strategy (like e-greedy). Use
        schedule_hyperparameters() for any hyperparameters that you want to change over time.

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        # Exploring (e-greedy update)
        if explore and random.random() < self.epsilon:
            return np.random.randint(self.action_space.n)
        # Greedy update
        else:
            actions = self.critics_net(torch.from_numpy(obs).float())
            a = torch.argmax(actions)
            return a.item()

    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN
        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your network and return the Q-loss in the form of a
        dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        # Grab expected Q values from the value nextwork of current states
        q_now = self.critics_net(batch.states).gather(1, batch.actions.long()) # Q(s_t, a_t; \theta)
        # Query target network for next state action values
        q_next = self.critics_target(batch.next_states).detach().max(1)[0].unsqueeze(1) # max a' Q(s_t+1, a'; \theta')
        # Compute targets of current states
        q_targets = batch.rewards + (self.gamma * (1 - batch.done) * q_next)
        # Compute loss
        q_loss = F.mse_loss(q_targets, q_now)

        # Optimize the model
        self.critics_optim.zero_grad()
        q_loss.backward()
        self.critics_optim.step()

        # Periodically change target weights
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.critics_target.hard_update(self.critics_net)
        return {"q_loss": q_loss.detach()}

class Reinforce(Agent):
    """ The Reinforce Agent for Ex 3
    :attr critics_net (FCNetwork): fully connected critic network to compute Q-value estimates
    :attr critics_target (FCNetwork): fully connected target critic network
    :attr critics_optim (torch.optim): PyTorch optimiser for critics network
    :attr actors_net (FCNetwork): fully connected actor network for policy
    :attr actors_optim (torch.optim): PyTorch optimiser for actor network
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        gamma: float,
        **kwargs,
    ):
        """
        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param gamma (float): discount rate gamma
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #

        ### DO NOT CHANGE THE OUTPUT ACTIVATION OF THIS POLICY ###
        self.policy = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=torch.nn.modules.activation.Softmax
        )

        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate, eps=0.001)

        # ############################################# #
        # WRITE ANY EXTRA HYPERPARAMETERS YOU NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.gamma = gamma

        # ############################### #
        # WRITE ANY AGENT PARAMETERS HERE #
        # ############################### #

        # ###############################################
        self.saveables.update(
            {
                "policy": self.policy,
            }
        )

    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Updates the hyperparameters 

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        pass

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        Select an action from the model's stochastic policy by sampling a discrete action
        from the distribution specified by the model output

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        # Sample policy for action probabilities
        probs = self.policy(torch.from_numpy(obs).float())
        # Sample action according to policy
        action = np.random.choice(range(self.action_space.n), p=probs.detach().numpy())

        if explore: # Training time
            return action
        else: # Greedy at eval time --> obey learnt policy
            return np.argmax(probs.detach().numpy())
        
    def update(
        self, rewards: List[float], observations: List[np.ndarray], actions: List[int],
        ) -> Dict[str, float]:
        """Update function for policy gradients

        :param rewards (List[float]): rewards of episode (from first to last)
        :param observations (List[np.ndarray]): observations of episode (from first to last)
        :param actions (List[int]): applied actions of episode (from first to last)
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
            losses
        """   
        # Storage of returns
        R = 0
        returns = []
        # Calculate discounted returns
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        # Calculate log probs from policy
        returns_tensor = torch.Tensor(returns).unsqueeze(1) # Conver to tensor for backprop
        s = torch.Tensor(observations)
        a = torch.Tensor(actions).unsqueeze(1)
        log_probs = -torch.log(self.policy(s).gather(1, a.long()))
        # Compute policy loss
        p_loss = (log_probs*returns_tensor).mean()

        # Optimize model
        p_loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy_optim.step()

        return {"p_loss": p_loss.detach().item()}