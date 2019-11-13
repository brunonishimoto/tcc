from dialogue_system.dm.agents.drqn_agent_new import DRQNAgentNew
import random
import dialogue_system.constants as const
import numpy as np

from utils.util import log


# Some of the code based off of https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
# Note: In original paper's code the epsilon is not annealed and annealing is not implemented in this code either


class DRQNSoftmaxNew(DRQNAgentNew):
    """The DQN agent that interacts with the user."""

    def __init__(self, params):
        """
        The constructor of DQNAgent.

        The constructor of DQNAgent which saves params, sets up neural network graphs, etc.

        Parameters:
            state_size (int): The state representation size or length of numpy array
            params (dict): Loaded params in dict

        """

        super().__init__(params)
        self.tau_init = self.C['tau_init']
        self.tau_stop = self.C['tau_stop']
        self.tau_decay = self.C['tau_decay']
        self.tau = self.tau_init

    def get_action(self, state, step=None, use_rule=False, train=True):
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

                # Linear decay
                a = -float(self.tau_init - self.tau_stop) / self.tau_decay
                b = float(self.tau_init)
                self.tau = max(self.tau_stop, a * float(step) + b)

                # Softmax

                q_values = self._dqn_predict(state.reshape(1, *self.state_size))
                q_modified = q_values / self.tau
                q_max = np.max(q_modified)
                exp_values = np.exp(q_modified - q_max)

                probs = exp_values / np.sum(exp_values)
                index = np.random.choice(list(range(self.num_actions)), p=probs.reshape(self.num_actions))

                action = self._map_index_to_action(index)
                return index, action
        else:
            return self._dqn_action(state)
