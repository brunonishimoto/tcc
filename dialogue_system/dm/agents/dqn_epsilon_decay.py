from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from dialogue_system.dm.agents.dqn_agent import DQNAgent
import dialogue_system.constants as const
import dialogue_system.dialogue_config as config
import random
import copy
import numpy as np
import re


# Some of the code based off of https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
# Note: In original paper's code the epsilon is not annealed and annealing is not implemented in this code either


class DQNEpsilonDecay(DQNAgent):
    """The DQN agent that interacts with the user."""

    def __init__(self, state_size, params):
        """
        The constructor of DQNAgent.

        The constructor of DQNAgent which saves params, sets up neural network graphs, etc.

        Parameters:
            state_size (int): The state representation size or length of numpy array
            params (dict): Loaded params in dict

        """

        super().__init__(state_size, params)
        self.eps_init = self.C['epsilon_init']
        self.eps_stop = self.C['epsilon_stop']
        self.decay_rate = self.C['decay_rate']
        self.explore_prob = self.eps_init

    def get_action(self, state, use_rule=False, train=True):
        """
        Returns the action of the agent given a state.

        Gets the action of the agent given the current state. Either the rule-based policy or the neural networks are
        used to respond.

        Parameters:
            state (numpy.array): The database with format dict(long: dict)
            use_rule (bool): Indicates whether or not to use the rule-based policy, which depends on if this was called
                             in warmup or training. Default: False

        Returns:
            int: The index of the action in the possible actions
            dict: The action/response itself

        """

        if train:
            if use_rule:
                return self._rule_action()
            else:
                self.num_steps += 1

                # Linear decay

                a = -float(self.eps_init - self.eps_stop) / self.decay_rate
                b = float(self.eps_init)
                self.explore_prob = max(self.eps_stop, a * float(self.num_steps) + b)

                if self.explore_prob > random.random():
                    index = np.argmax(np.random.rand(self.num_actions))

                    action = self._map_index_to_action(index)
                    return index, action
                else:
                    return self._dqn_action(state)
        else:
            return self._dqn_action(state, train=False)
