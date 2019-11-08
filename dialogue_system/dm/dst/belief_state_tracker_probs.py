from dialogue_system.dm.dst.db_query import DBQuery
from utils.util import convert_list_to_dict, remove_empty_slots, log
import dialogue_system.dialogue_config as cfg
import dialogue_system.constants as const
import numpy as np
import copy
import pickle
import random


class BeliefStateTrackerProbs:
    """Tracks the state of the episode/conversation and prepares the state representation for the agent."""

    def __init__(self, config):
        """
        The constructor of StateTracker.

        The constructor of StateTracker which creates a DB query object, creates necessary state rep. dicts, etc. and
        calls reset.

        Parameters:
            config (dict): Loaded config in dict

        """

        # Load movie DB
        # Note: If you get an unpickling error here then run 'pickle_converter.py' and it should fix it
        database_path = config['db_file_paths']['database']
        database = pickle.load(open(database_path, 'rb'), encoding='latin1')

        # Values for confusing a user action
        dict_path = config['db_file_paths']['dict']
        self.movie_dict = pickle.load(open(dict_path, 'rb'), encoding='latin1')

        self.n_best = config['dst']['n_best']
        self.num_sequences = config['dst']['num_sequences']
        self.intent_error_prob = config['dst']['intent_error_prob']
        self.value_error_prob = config['dst']['value_error_prob']
        self.slot_error_prob = config['dst']['slot_error_prob']
        self.miss_error_prob = config['dst']['miss_error_prob']

        # Clean DB
        remove_empty_slots(database)

        self.db_helper = DBQuery(database)
        self.match_key = cfg.usersim_default_key
        self.intents_dict = convert_list_to_dict(cfg.all_intents)
        self.num_intents = len(cfg.all_intents)
        self.slots_dict = convert_list_to_dict(cfg.all_slots)
        self.num_slots = len(cfg.all_slots)
        self.max_round_num = config['run']['max_round_num']
        self.none_state = np.zeros(self.get_state_size())


        self.reset()

    def get_state_size(self):
        """Returns the state size of the state representation used by the agent."""

        return (self.num_sequences, self.n_best * (2 * self.num_intents + 7 * self.num_slots + 3 + self.max_round_num + 1))

    def reset(self):
        """Resets current_informs, history and round_num."""

        self.current_informs = []
        self.history = []
        for i in range(self.n_best):
            self.current_informs.append({})
        # A list of the dialogues (dicts) by the agent and user so far in the conversation
        self.round_num = 0
        self.history_states = np.zeros(self.get_state_size())

    def print_history(self):
        """Helper function if you want to see the current history action by action."""

        for action in self.history:
            print(action)

    def get_suggest_slots_values(self, request_slots):
        """ Get the suggested values for request slots """

        suggest_slot_vals = {}
        if len(request_slots) > 0:
            suggest_slot_vals = self.db_helper.suggest_slot_values(request_slots, self.current_informs[0])

        return suggest_slot_vals

    def get_current_kb_results(self):
        """ get the kb_results for current state """
        kb_results = self.db_helper.get_db_results(self.current_informs[0])
        return kb_results

    def get_state(self, done=False):
        """
        Returns the state representation as a numpy array which is fed into the agent's neural network.

        The state representation contains useful information for the agent about the current state of the conversation.
        Processes by the agent to be fed into the neural network. Ripe for experimentation and optimization.

        Parameters:
            done (bool): Indicates whether this is the last dialogue in the episode/conversation. Default: False

        Returns:
            numpy.array: A numpy array of shape (state size,)

        """

        # If done then fill state with zeros
        if done:
            return self.none_state

        # Representations that use user action
        n_best_last_user_action = self.history[-1]

        # Create one-hot of intents to represent the current user action
        user_act_rep = np.zeros((self.n_best, self.num_intents))

        # Create bag of inform slots representation to represent the current user action
        user_inform_slots_rep = np.zeros((self.n_best, self.num_slots))

        # Create bag of request slots representation to represent the current user action
        user_request_slots_rep = np.zeros((self.n_best, self.num_slots))

        # Create bag of filled_in slots based on the current_slots
        current_slots_rep = np.zeros((self.n_best, self.num_slots))

        # Representations that use agent action
        last_agent_action = self.history[-2] if len(self.history) > 1 else None

        # Encode last agent intent
        agent_act_rep = np.zeros((self.n_best, self.num_intents))

        # Encode last agent inform slots
        agent_inform_slots_rep = np.zeros((self.n_best, self.num_slots))

        # Encode last agent request slots
        agent_request_slots_rep = np.zeros((self.n_best, self.num_slots))

        # Value representation of the round num
        turn_rep = np.zeros((self.n_best, 1)) + self.round_num / 5.

        # One-hot representation of the round num
        turn_onehot_rep = np.zeros((self.n_best, self.max_round_num))

        # Representations that use db
        db_results_dict = []

        # Representation of DB query results (scaled counts)
        kb_count_rep = np.zeros((self.n_best, self.num_slots + 1))

        # Representation of DB query results (binary)
        kb_binary_rep = np.zeros((self.n_best, self.num_slots + 1))

        # Representation of probabilities
        probs_rep = np.zeros((self.n_best, 1))

        for i in range(self.n_best):
            user_act_rep[i][self.intents_dict[n_best_last_user_action[i]['action'][const.INTENT]]] = 1.0

            for key in n_best_last_user_action[i]['action'][const.INFORM_SLOTS].keys():
                user_inform_slots_rep[i][self.slots_dict[key]] = 1.0

            for key in n_best_last_user_action[i]['action'][const.REQUEST_SLOTS].keys():
                user_request_slots_rep[i][self.slots_dict[key]] = 1.0

            for key in self.current_informs[i]:
                current_slots_rep[i][self.slots_dict[key]] = 1.0 * self.current_informs[i][key]['prob']

            if last_agent_action:
                agent_act_rep[i][self.intents_dict[last_agent_action[const.INTENT]]] = 1.0

                for key in last_agent_action[const.INFORM_SLOTS].keys():
                    agent_inform_slots_rep[i][self.slots_dict[key]] = 1.0

                for key in last_agent_action[const.REQUEST_SLOTS].keys():
                    agent_request_slots_rep[i][self.slots_dict[key]] = 1.0

            turn_onehot_rep[i][self.round_num - 1] = 1.0

            constraints = {}
            for k in self.current_informs[i]:
                constraints[k] = self.current_informs[i][k]['value']

            db_results_dict.append(self.db_helper.get_db_results_for_slots(constraints))

            kb_count_rep[i] = db_results_dict[i][const.KB_MATCHING_ALL_CONSTRAINTS] / 100.
            for key in db_results_dict[i].keys():
                if key in self.slots_dict:
                    kb_count_rep[i][self.slots_dict[key]] = db_results_dict[i][key] / 100.

            kb_binary_rep[i] = np.sum(db_results_dict[i][const.KB_MATCHING_ALL_CONSTRAINTS] > 0.)
            for key in db_results_dict[i].keys():
                if key in self.slots_dict:
                    kb_binary_rep[i][self.slots_dict[key]] = np.sum(db_results_dict[i][key] > 0.)

            masked_kb_count = np.ma.compressed(np.ma.masked_where(self.current_informs[i] > 0, kb_count_rep[i]))
            probs_rep[i] = n_best_last_user_action[i]['prob'] *

        state_representation = np.hstack(
            [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
             agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep,
             kb_count_rep, probs_rep]).flatten()

        self.history_states = np.roll(self.history_states, -1, axis=0)
        self.history_states[-1] = state_representation
        return self.history_states

    def update_state_agent(self, agent_action):
        """
        Updates the dialogue history with the agent's action and augments the agent's action.

        Takes an agent action and updates the history. Also augments the agent_action param with query information and
        any other necessary information.

        Parameters:
            agent_action (dict): The agent action of format dict('intent': '', 'inform_slots': {},
                                 'request_slots': {}) and changed to dict('intent': '', 'inform_slots': {},
                                 'request_slots': {}, 'round': int, 'speaker': 'Agent')
        """

        if agent_action[const.INTENT] == const.INFORM:
            assert agent_action[const.INFORM_SLOTS]

            constraints = {}
            for k in self.current_informs[0]:
                constraints[k] = self.current_informs[0][k]['value']

            inform_slots = self.db_helper.fill_inform_slot(agent_action[const.INFORM_SLOTS], constraints)
            agent_action[const.INFORM_SLOTS] = inform_slots
            assert agent_action[const.INFORM_SLOTS]
            for key, value in list(agent_action[const.INFORM_SLOTS].items()):
                assert key != const.MATCH_FOUND
                assert value != const.PLACEHOLDER, 'KEY: {}'.format(key)
                for i in range(self.n_best):
                    self.current_informs[i][key] = {'value': value, 'prob': 1.0}
        # If intent is match_found then fill the action informs with the matches informs (if there is a match)
        elif agent_action[const.INTENT] == const.MATCH_FOUND:
            assert not agent_action[const.INFORM_SLOTS], 'Cannot inform and have intent of match found!'

            constraints = {}
            for k in self.current_informs[0]:
                constraints[k] = self.current_informs[0][k]['value']

            db_results = self.db_helper.get_db_results(constraints)
            if db_results:
                # Arbitrarily pick the first value of the dict
                key, value = list(db_results.items())[0]
                agent_action[const.INFORM_SLOTS] = copy.deepcopy(value)
                agent_action[const.INFORM_SLOTS][self.match_key] = str(key)
            else:
                agent_action[const.INFORM_SLOTS][self.match_key] = const.NO_MATCH
            for i in range(self.n_best):
                self.current_informs[i][self.match_key] = {'value': agent_action[const.INFORM_SLOTS][self.match_key], 'prob': 1.0}
        agent_action.update({const.ROUND: self.round_num, const.SPEAKER_TYPE: const.AGT_SPEAKER_VAL})
        self.history.append(agent_action)

    def update_state_user(self, user_action):
        """
        Updates the dialogue history with the user's action and augments the user's action.

        Takes a user action and updates the history. Also augments the user_action param with necessary information.

        Parameters:
            user_action (dict): The user action of format dict('intent': '', 'inform_slots': {},
                                 'request_slots': {}) and changed to dict('intent': '', 'inform_slots': {},
                                 'request_slots': {}, 'round': int, 'speaker': 'User')

        """

        n_best_actions = self.__generate_noise_user_actions(user_action)
        for idx, action in enumerate(n_best_actions):
            for key, value in action['action'][const.INFORM_SLOTS].items():
                self.current_informs[idx][key] = {'value': value, 'prob': action['prob']}
        user_action.update({const.ROUND: self.round_num + 1, const.SPEAKER_TYPE: const.USR_SPEAKER_VAL})
        self.history.append(n_best_actions)
        self.round_num += 1

    def __generate_noise_user_actions(self, user_action):
        intent = user_action[const.INTENT]
        inform_slots = user_action[const.INFORM_SLOTS]
        request_slots = user_action[const.REQUEST_SLOTS]

        n_best_confused_actions = []
        n_best_confused_actions.append({'action': user_action, 'prob': 1 / self.n_best})
        for i in range(1, self.n_best):
            n_best_confused_actions.append({'action': self.__create_wrong_action(user_action), 'prob': 1 / self.n_best})

        return n_best_confused_actions

    def __create_wrong_action(self, user_action):
        action = copy.deepcopy(user_action)

        inform_slots = copy.deepcopy(list(action[const.INFORM_SLOTS]))
        request_slots = copy.deepcopy(list(action[const.REQUEST_SLOTS]))

        if inform_slots and request_slots:
            if np.random.random() < 0.7:
                action = self.__confuse_inform(action)
            else:
                action = self.__confuse_request(action)
        elif inform_slots:
                action = self.__confuse_inform(action)
        elif request_slots:
                action = self.__confuse_request(action)
        else:
            action = self.__confuse_intent(action)

        # Confuse intent with the probability
        if np.random.random() < self.intent_error_prob:
            action = self.__confuse_intent(action)

        return action

    def __confuse_intent(self, action):
        # Error on intent
        possible_intents = copy.copy(cfg.user_intents)
        possible_intents.remove(action[const.INTENT])
        # if action[const.INTENT] != const.THANKS:
        #     possible_intents.remove(const.THANKS)

        # action[const.INTENT] = np.random.choice(possible_intents)

        # possible_slots = copy.copy(cfg.all_slots)
        # possible_slots.remove(cfg.usersim_default_key)
        # if action[const.INTENT] == const.INFORM:
        #     # Sample a random slot-value pair
        #     slot = np.random.choice(possible_slots)
        #     action[const.INFORM_SLOTS][slot] = np.random.choice(self.movie_dict[slot])
        # elif action[const.INTENT] == const.REQUEST:
        #     # Sample a random slot to request
        #     slot = np.random.choice(possible_slots)
        #     action[const.REQUEST_SLOTS][slot] = const.UNKNOWN
        # else:
        #     action[const.INFORM_SLOTS] = {}
        #     action[const.REQUEST_SLOTS] = {}
        action[const.INTENT] = np.random.choice(possible_intents)
        return action

    def __confuse_inform(self, action):
        # 0 - only value, 1 - slot and value, 2 - add a random slot and value
        choice = np.random.choice([0, 1, 2], p=[self.value_error_prob, self.slot_error_prob, self.miss_error_prob])

        if choice == 0:
            # Choose the slot to be changed
            slot = np.random.choice(list(action[const.INFORM_SLOTS]))
            new_value = np.random.choice(self.movie_dict[slot])
            action[const.INFORM_SLOTS][slot] = new_value
        elif choice == 1:
            # Change a inform slot
            possible_slots = copy.copy(cfg.all_slots)

            # Choose and remove the slot to be changed
            slot = np.random.choice(list(action[const.INFORM_SLOTS]))
            action[const.INFORM_SLOTS].pop(slot)

            possible_slots.remove(slot)

            if cfg.usersim_default_key in possible_slots:
                possible_slots.remove(cfg.usersim_default_key)

            # Select a new slot and its value
            new_slot = np.random.choice(possible_slots)
            new_value = np.random.choice(self.movie_dict[new_slot])
            action[const.INFORM_SLOTS][new_slot] = new_value
        elif choice == 2:
            # Add a random slot-value pair
            possible_slots = copy.copy(cfg.all_slots)

            # remove from possible slots the ones in it
            for slot in list(action[const.INFORM_SLOTS]):
                possible_slots.remove(slot)
            if cfg.usersim_default_key in possible_slots:
                possible_slots.remove(cfg.usersim_default_key)

            # Select a new slot and its value
            new_slot = np.random.choice(possible_slots)
            new_value = np.random.choice(self.movie_dict[new_slot])
            action[const.INFORM_SLOTS][new_slot] = new_value

        return action

    def __confuse_request(self, action):
        # Change a request slot
        possible_slots = copy.copy(cfg.all_slots)

        # Choose and remove the slot to be changed
        slot = np.random.choice(list(action[const.REQUEST_SLOTS]))
        action[const.REQUEST_SLOTS].pop(slot)

        possible_slots.remove(slot)

        # Select a new slot
        new_slot = np.random.choice(possible_slots)
        action[const.REQUEST_SLOTS][new_slot] = const.UNKNOWN

        return action
