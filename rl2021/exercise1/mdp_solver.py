from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Optional, Hashable

from rl2021.utils import MDP, Transition, State, Action

class MDPSolver(ABC):
    """Base class for MDP solvers
    :attr mdp (MDP): MDP to solve
    :attr gamma (float): discount factor gamma to use
    :attr action_dim (int): number of actions in the MDP
    :attr state_dim (int): number of states in the MDP
    """

    def __init__(self, mdp: MDP, gamma: float):
        """Constructor of MDPSolver

        Initialises some variables from the MDP, namely the state and action dimension variables
        
        :param mdp (MDP): MDP to solve
        :param gamma (float): discount factor (gamma)
        """
        self.mdp: MDP = mdp
        self.gamma: float = gamma

        self.action_dim: int = len(self.mdp.actions)
        self.state_dim: int = len(self.mdp.states)

    def decode_policy(self, policy: Dict[int, np.ndarray]) -> Dict[State, Action]:
        """Generates greedy, deterministic policy dict

        Given a stochastic policy from state indices to distribution over actions, the greedy,
        deterministic policy is generated choosing the action with highest probability

        :param policy (Dict[int, np.ndarray of float with dim (num of actions)]):
            stochastic policy assigning a distribution over actions to each state index
        :return (Dict[State, Action]): greedy, deterministic policy from states to actions
        """
        new_p = {}
        for state, state_idx in self.mdp._state_dict.items():
            new_p[state] = self.mdp.actions[np.argmax(policy[state_idx])]
        return new_p

    @abstractmethod
    def solve(self):
        """Solves the given MDP
        """
        ...


class ValueIteration(MDPSolver):
    """MDP solver using the Value Iteration algorithm
    """

    def _one_step_lookahead(self, state: int, V: np.ndarray) -> np.ndarray:
        """
            Calculates the values associated with a one step lookahead
            from a specified state.

            :param state (int): integer reprenting the current state of the MDP.
            :param: V (np.ndarray): array of current state values.
            :return action_values (np.ndarray): array of values associated with
                                                states that can be reached via 
                                                one step lookahead.
        """
        next_state_vals = np.zeros(self.action_dim)
        # Consider every possible state action pair from state s
        for action in range(self.action_dim):
            for next_state in range(self.state_dim):
                # Update values given by Bellman equation
                next_state_vals[action] += self.mdp.P[state, action, next_state] * \
                                           (self.mdp.R[state, action, next_state] + self.gamma * V[next_state])
        return next_state_vals

    def _calc_value_func(self, theta: float) -> np.ndarray:
        """Calculates the value function
        Useful Variables:
        1. `self.mdp` -- Gives access to the MDP.
        2. `self.mdp.R` -- 3D NumPy array with the rewards for each transition.
            E.g. the reward of transition [3] -2-> [4] (going from state 3 to state 4 with action
            2) can be accessed with `self.R[3, 2, 4]`
        3. `self.mdp.P` -- 3D NumPy array with transition probabilities.
            *REMEMBER*: the sum of (STATE, ACTION, :) should be 1.0 (all actions lead somewhere)
            E.g. the transition probability of transition [3] -2-> [4] (going from state 3 to
            state 4 with action 2) can be accessed with `self.P[3, 2, 4]`

        :param theta (float): theta is the stop threshold for value iteration
        :return (np.ndarray of float with dim (num of states)):
            1D NumPy array with the values of each state.
            E.g. V[3] returns the computed value for state 3
        """
        # Initialize value function as 0 everywhere
        V = np.zeros(self.state_dim)
        converged = False 
        while not converged:
            delta = 0
            # Update each state
            for state in range(self.state_dim):
                # Store old state value
                v = V[state]
                # Greedy update of state values via one step lookahead
                next_state_vals = self._one_step_lookahead(state, V)
                V[state] = np.amax(next_state_vals)
                delta = np.maximum(delta, np.abs(v - V[state]))
            # Convergence check
            if delta < theta:
                converged = True
        return V

    def _calc_policy(self, V: np.ndarray) -> np.ndarray:
        """Calculates the policy

        :param V (np.ndarray of float with dim (num of states)):
            A 1D NumPy array that encodes the computed value function (from _calc_value_func(...))
            It is indexed as (State) where V[State] is the value of state 'State'
        :return (np.ndarray of float with dim (num of states, num of actions):
            A 2D NumPy array that encodes the calculated policy.
            It is indexed as (STATE, ACTION) where policy[STATE, ACTION] has the probability of
            taking action 'ACTION' in state 'STATE'.
            REMEMBER: the sum of policy[STATE, :] should always be 1.0
            For deterministic policies the following holds for each state S:
            policy[S, BEST_ACTION] = 1.0
            policy[S, OTHER_ACTIONS] = 0
        """
        policy = np.zeros([self.state_dim, self.action_dim])
        for state in range(self.state_dim):
            # Greedy action selection in state s
            best_action = np.argmax(self._one_step_lookahead(state, V))
            # Greedy policy update
            policy[state, best_action] = 1
        return policy

    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        Compiles the MDP and then calls the calc_value_func and
        calc_policy functions to return the best policy and the
        computed value function

        **DO NOT CHANGE THIS FUNCTION**

        :param theta (float, optional): stop threshold, defaults to 1e-6
        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)):
            Tuple of calculated policy and value function
        """
        self.mdp.ensure_compiled()
        V = self._calc_value_func(theta)
        policy = self._calc_policy(V)

        return policy, V


class PolicyIteration(MDPSolver):
    """MDP solver using the Policy Iteration algorithm
    """

    def _one_step_lookahead(self, state: int, V: np.ndarray) -> np.ndarray:
        """
            Calculates the values associated with a one step lookahead
            from a specified state.

            :param state (int): integer reprenting the current state of the MDP.
            :param: V (np.ndarray): array of current state values.
            :return action_values (np.ndarray): array of values associated with
                                                states that can be reached via 
                                                one step lookahead.
        """
        next_state_vals = np.zeros(self.action_dim)
        # Consider every possible state action pair from state s
        for action in range(self.action_dim):
            for next_state in range(self.state_dim):
                # Update values given by Bellman equation
                next_state_vals[action] += self.mdp.P[state, action, next_state] * \
                                           (self.mdp.R[state, action, next_state] + self.gamma * V[next_state])
        return next_state_vals
        
    def _policy_eval(self, policy: np.ndarray) -> np.ndarray:
        """Computes one policy evaluation step
        :param policy (np.ndarray of float with dim (num of states, num of actions)):
            A 2D NumPy array that encodes the policy.
            It is indexed as (STATE, ACTION) where policy[STATE, ACTION] has the probability of
            taking action 'ACTION' in state 'STATE'.
            REMEMBER: the sum of policy[STATE, :] should always be 1.0
            For deterministic policies the following holds for each state S:
            policy[S, BEST_ACTION] = 1.0
            policy[S, OTHER_ACTIONS] = 0
        :return (np.ndarray of float with dim (num of states)): 
            A 1D NumPy array that encodes the computed value function
            It is indexed as (State) where V[State] is the value of state 'State'
        """

        # Initialize value function as 0 everywhere
        V = np.zeros(self.state_dim)
        converged = False 
        while not converged:
            delta = 0
            # Update each state
            for state in range(self.state_dim):
                # Store old state value
                v = 0 
                # Try all possible actions from state
                for action, action_prob in enumerate(policy[state]):
                    for next_state in range(self.state_dim):
                        # Update values given by Bellman equation
                        v += action_prob * self.mdp.P[state, action, next_state] * \
                                    (self.mdp.R[state, action, next_state] + self.gamma * V[next_state])
                delta = np.maximum(delta, np.abs(v - V[state]))
                V[state] = v
            # Convergence check
            if delta < self.theta:
                converged = True
        return np.array(V)

    def _policy_improvement(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes one policy improvement iteration

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        Useful Variables (As with Value Iteration):
        1. `self.mpd` -- Gives access to the MDP.
        2. `self.mdp.R` -- 3D NumPy array with the rewards for each transition.
            E.g. the reward of transition [3] -2-> [4] (going from state 3 to state 4 with action
            2) can be accessed with `self.R[3, 2, 4]`
        3. `self.mdp.P` -- 3D NumPy array with transition probabilities.
            *REMEMBER*: the sum of (STATE, ACTION, :) should be 1.0 (all actions lead somewhere)
            E.g. the transition probability of transition [3] -2-> [4] (going from state 3 to
            state 4 with action 2) can be accessed with `self.P[3, 2, 4]`

        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)):
            Tuple of calculated policy and value function
        """
        # Initialize arbitrary policy
        policy = np.zeros([self.state_dim, self.action_dim])
        # Initialize arbitrary state values
        V = np.zeros([self.state_dim])
        while True:
            # 1. Policy evaluation
            V = self._policy_eval(policy)

            # Did the policy change after updating it?
            stable_policy = True
            # Store old policy for comparison
            old_policy = policy.copy()
            
            # 2. Policy improvement
            for state in range(self.state_dim):
                # Perform one step lookahead
                next_state_vals = self._one_step_lookahead(state, V)
                # Greedy policy update
                policy[state] = np.zeros(self.action_dim)
                policy[state][np.argmax(next_state_vals)] = 1

                # Have we reached convergence?
                if not np.all(old_policy[state] == policy[state]):
                    stable_policy = False

            if stable_policy:
                return policy, V

    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        This function compiles the MDP and then calls the
        policy improvement function that the student must implement
        and returns the solution

        **DO NOT CHANGE THIS FUNCTION**

        :param theta (float, optional): stop threshold, defaults to 1e-6
        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)]):
            Tuple of calculated policy and value function
        """
        self.mdp.ensure_compiled()
        self.theta = theta
        return self._policy_improvement()


if __name__ == "__main__":
    mdp = MDP()
    mdp.add_transition(
        #          state   action next_state prob reward
        Transition("high", "wait", "high", 1, 2),
        Transition("high", "search", "high", 0.8, 5),
        Transition("high", "search", "low", 0.2, 5),
        Transition("high", "recharge", "high", 1, 0),
        Transition("low", "recharge", "high", 1, 0),
        Transition("low", "wait", "low", 1, 2),
        Transition("low", "search", "high", 0.6, -3),
        Transition("low", "search", "low", 0.4, 5),
    )

    solver = ValueIteration(mdp, 0.9)
    policy, valuefunc = solver.solve()
    print("---Value Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function")
    print(valuefunc)

    solver = PolicyIteration(mdp, 0.9)
    policy, valuefunc = solver.solve()
    print("---Policy Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function")
    print(valuefunc)