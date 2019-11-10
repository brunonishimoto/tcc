import dialogue_system.dm.agents.models as models
import dialogue_system.constants as const
import dialogue_system.dialogue_config as cfg
import random
import copy
import numpy as np
import re
import os


# Some of the code based off of https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
# Note: In original paper's code the epsilon is not annealed and annealing is not implemented in this code either


class DRQNAgent:
    """The DQN agent that interacts with the user."""

    def __init__(self, config):
        """
        The constructor of DQNAgent.

        The constructor of DQNAgent which saves config, sets up neural network graphs, etc.

        Parameters:
            state_size (int): The state representation size or length of numpy array
            config (dict): Loaded config in dict

        """
        self.C = config['agent']
        self.memory = []
        self.memory_index = 0
        self.max_memory_size = self.C['max_mem_size']
        self.vanilla = self.C['vanilla']
        self.gamma = self.C['gamma']
        self.batch_size = self.C['batch_size']
        self.trace_length = self.C['trace_length']
        self.db_attention = self.C['db_attention']

        self.load_weights_file_path = self.C['load_weights_file_path']
        self.save_weights_file_path = self.C['save_weights_file_path']

        # Create directory if it does not exist
        if self.save_weights_file_path and not os.path.exists(os.path.split(self.save_weights_file_path)[0]):
            os.makedirs(os.path.split(self.save_weights_file_path)[0])

        self.len_mem = 0
        if self.max_memory_size < self.batch_size:
            raise ValueError('Max memory size must be at least as great as batch size!')

        self.possible_actions = cfg.agent_actions
        self.num_actions = len(self.possible_actions)

        self.action_counts = np.ones(self.num_actions)

        self.rule_request_set = cfg.rule_requests
        self.rule_inform_set = cfg.rule_informs

        self.reset()

    def build_models(self, state_size, db_size=None):
        self.state_size = state_size
        self.C['model']['input_dim'] = state_size
        self.C['model']['output_dim'] = self.num_actions
        self.C['model']['db_size'] = db_size
        self.db_size = db_size

        self.beh_model = models.load(self.C).build_model()
        self.tar_model = models.load(self.C).build_model()

        self.__load_weights()


    def reset(self):
        """Resets the rule-based variables."""
        self.rule_current_request_slot_index = 0
        self.rule_current_inform_slot_index = 0
        self.first_turn = True
        self.rule_phase = const.NOT_DONE

    def get_action(self, state, step=None, use_rule=False, train=True):
        # Implemented in child classes
        pass

    def _rule_action(self):
        """
        Returns a rule-based policy action.

        Selects the next action of a simple rule-based policy.

        Returns:
            int: The index of the action in the possible actions
            dict: The action/response itself

        """
        if self.first_turn:
            self.first_turn = False

            rule_response =  {const.INTENT: const.GREETING, const.INFORM_SLOTS: {},
                            const.REQUEST_SLOTS: {}}
        elif self.rule_current_request_slot_index < len(self.rule_request_set):
            slot = self.rule_request_set[self.rule_current_request_slot_index]
            self.rule_current_request_slot_index += 1

            rule_response = {const.INTENT: const.REQUEST, const.INFORM_SLOTS: {},
                             const.REQUEST_SLOTS: {slot: const.UNKNOWN}}
        elif self.rule_current_inform_slot_index < len(self.rule_inform_set):
            slot = self.rule_inform_set[self.rule_current_inform_slot_index]
            self.rule_current_inform_slot_index += 1

            rule_response = {const.INTENT: const.INFORM, const.INFORM_SLOTS: {slot: const.PLACEHOLDER},
                             const.REQUEST_SLOTS: {}}
        elif self.rule_phase == const.NOT_DONE:
            rule_response = {const.INTENT: const.MATCH_FOUND, const.INFORM_SLOTS: {}, const.REQUEST_SLOTS: {}}
            self.rule_phase = const.DONE
        elif self.rule_phase == const.DONE:
            rule_response = {const.INTENT: const.THANKS, const.INFORM_SLOTS: {}, const.REQUEST_SLOTS: {}}
        else:
            raise Exception('Should not have reached this clause')

        index = self._map_action_to_index(rule_response)
        return index, rule_response

    def _map_action_to_index(self, response):
        """
        Maps an action to an index from possible actions.

        Parameters:
            response (dict)

        Returns:
            int
        """

        for (i, action) in enumerate(self.possible_actions):
            if response == action:
                return i
        raise ValueError(f'Response: {response} not found in possible actions')

    def _dqn_action(self, state):
        """
        Returns a behavior model output given a state.

        Parameters:
            state (numpy.array)

        Returns:
            int: The index of the action in the possible actions
            dict: The action/response itself
        """
        index = np.argmax(self._dqn_predict_one(state))

        action = self._map_index_to_action(index)
        return index, action

    def _map_index_to_action(self, index):
        """
        Maps an index to an action in possible actions.

        Parameters:
            index (int)

        Returns:
            dict
        """

        for (i, action) in enumerate(self.possible_actions):
            if index == i:
                return copy.deepcopy(action)
        raise ValueError('Index: {} not in range of possible actions'.format(index))

    def _dqn_predict_one(self, state, target=False):
        """
        Returns a model prediction given a state.

        Parameters:
            state (numpy.array)
            target (bool)

        Returns:
            numpy.array
        """

        return self._dqn_predict(state.reshape(1, *self.state_size), target=target).flatten()

    def _dqn_predict(self, states, target=False):
        """
        Returns a model prediction given an array of states.

        Parameters:
            states (numpy.array)
            target (bool)

        Returns:
            numpy.array
        """

        if target:
            if not self.db_attention:
                return self.tar_model.predict(states)
            else:
                reshaped = np.reshape(states[0], (self.state_size[0], *(5, self.state_size[1] // 5)))
                observation = reshaped[:, :, :-(self.db_size[1] // 5)]
                observation = observation.reshape((observation.shape[0], observation.shape[1] * observation.shape[2]))
                observation = observation.reshape(1, *observation.shape)
                db_input = reshaped[:, :, -(self.db_size[1] // 5):]
                db_input = db_input.reshape((db_input.shape[0], db_input.shape[1] * db_input.shape[2]))
                db_input = db_input.reshape(1, *db_input.shape)
                return self.tar_model.predict([observation, db_input])
        else:
            if not self.db_attention:
                return self.beh_model.predict(states)
            else:
                reshaped = np.reshape(states[0], (self.state_size[0], *(5, self.state_size[1] // 5)))
                observation = reshaped[:, :, :-(self.db_size[1] // 5)]
                observation = observation.reshape((observation.shape[0], observation.shape[1] * observation.shape[2]))
                observation = observation.reshape(1, *observation.shape)
                db_input = reshaped[:, :, -(self.db_size[1] // 5):]
                db_input = db_input.reshape((db_input.shape[0], db_input.shape[1] * db_input.shape[2]))
                db_input = db_input.reshape(1, *db_input.shape)
                return self.beh_model.predict([observation, db_input])

    def add_experience(self, episode):
        """
        Adds an experience tuple made of the parameters to the memory.

        Parameters:
            episode: array with:
                state (numpy.array)
                action (int)
                reward (int)
                next_state (numpy.array)
                done (bool)

        """

        if len(self.memory) < self.max_memory_size:
            self.memory.append(None)
        self.memory[self.memory_index] = episode
        self.memory_index = (self.memory_index + 1) % self.max_memory_size
        self.len_mem += len(episode)

    def empty_memory(self):
        """Empties the memory and resets the memory index."""

        self.memory = []
        self.memory_index = 0

    def is_memory_full(self):
        """Returns true if the memory is full."""

        return self.len_mem >= self.max_memory_size

    def train(self):
        """
        Trains the agent by improving the behavior model given the memory tuples.

        Takes batches of memories from the memory pool and processing them. The processing takes the tuples and stacks
        them in the correct format for the neural network and calculates the Bellman equation for Q-Learning.

        """
        # Calc. num of batches to run
        num_batches = len(self.memory) // self.batch_size
        for b in range(num_batches):
            batch = random.sample(self.memory, self.batch_size)
            traces = []
            for episode in batch:
                while len(episode) < self.trace_length:
                    episode.append((np.zeros(batch[0][0][0].shape), 0, 0, np.zeros(batch[0][0][0].shape), 0))
                start_point = np.random.randint(0, len(episode) + 1 - self.trace_length)
                traces.append(episode[start_point:start_point + self.trace_length])

            traces = np.array(traces)
            batch = np.reshape(traces, [self.batch_size * self.trace_length, 5])

            states = np.array([sample[0] for sample in batch])
            next_states = np.array([sample[3] for sample in batch])

            states = np.resize(states, (self.batch_size * self.trace_length, *self.state_size))
            next_states = np.resize(next_states, (self.batch_size * self.trace_length, *self.state_size))

            assert states.shape == (self.batch_size * self.trace_length, *self.state_size), 'States Shape: {}'.format(states.shape)
            assert next_states.shape == states.shape

            beh_state_preds = self._dqn_predict(states)  # For leveling error
            if not self.vanilla:
                beh_next_states_preds = self._dqn_predict(next_states)  # For indexing for DDQN
            else:
                tar_next_state_preds = self._dqn_predict(next_states, target=True)  # For target value for DQN (& DDQN)

            inputs = np.zeros((self.batch_size * self.trace_length, *self.state_size))
            targets = np.zeros((self.batch_size * self.trace_length, self.num_actions))

            for i, (s, a, r, s_, d) in enumerate(batch):
                t = beh_state_preds[i]
                if not self.vanilla:
                    t[a] = r + self.gamma * tar_next_state_preds[i][np.argmax(beh_next_states_preds[i])] * (not d)
                else:
                    t[a] = r + self.gamma * np.amax(tar_next_state_preds[i]) * (not d)

                inputs[i] = s
                targets[i] = t

            self.beh_model.fit(inputs, targets, epochs=1, verbose=0)

    def copy(self):
        """Copies the behavior model's weights into the target model's weights."""

        self.tar_model.set_weights(self.beh_model.get_weights())

    def save_weights(self):
        """Saves the weights of both models in two h5 files."""

        if not self.save_weights_file_path:
            return
        beh_save_file_path = re.sub(r'\.h5', rf'_beh.h5', self.save_weights_file_path)
        self.beh_model.save_weights(beh_save_file_path)
        tar_save_file_path = re.sub(r'\.h5', rf'_tar.h5', self.save_weights_file_path)
        self.tar_model.save_weights(tar_save_file_path)

    def __load_weights(self):
        """Loads the weights of both models from two h5 files."""

        if self.load_weights_file_path:
            beh_load_file_path = re.sub(r'\.h5', r'_beh.h5', self.load_weights_file_path)
            self.beh_model.load_weights(beh_load_file_path)
            tar_load_file_path = re.sub(r'\.h5', r'_tar.h5', self.load_weights_file_path)
            self.tar_model.load_weights(tar_load_file_path)
