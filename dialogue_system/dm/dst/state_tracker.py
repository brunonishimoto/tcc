from dialogue_system.dm.dst.db_query import DBQuery
from utils.util import convert_list_to_dict, remove_empty_slots, log
import dialogue_system.dialogue_config as cfg
import dialogue_system.constants as const
import numpy as np
import copy
import pickle


class StateTracker:
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

        return 2 * self.num_intents + 7 * self.num_slots + 3 + self.max_round_num

    def reset(self):
        """Resets current_informs, history and round_num."""

        self.current_informs = {}
        # A list of the dialogues (dicts) by the agent and user so far in the conversation
        self.history = []
        self.round_num = 0

    def print_history(self):
        """Helper function if you want to see the current history action by action."""

        for action in self.history:
            print(action)

    def get_suggest_slots_values(self, request_slots):
        """ Get the suggested values for request slots """

        suggest_slot_vals = {}
        if len(request_slots) > 0:
            suggest_slot_vals = self.db_helper.suggest_slot_values(request_slots, self.current_informs)

        return suggest_slot_vals

    def get_current_kb_results(self):
        """ get the kb_results for current state """
        kb_results = self.db_helper.get_db_results(self.current_informs)
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

        user_action = self.history[-1]
        db_results_dict = self.db_helper.get_db_results_for_slots(self.current_informs)
        db_results = self.db_helper.get_db_results(self.current_informs)
        list_results = []
        for idx in list(db_results):
            list_results.append(db_results[idx])
        log(['dialogue'], f"DB results: {list_results}")
        log(['dialogue'], f"DB count: {db_results_dict}")
        log(['dialogue'], f"Current informs: {self.current_informs}")
        last_agent_action = self.history[-2] if len(self.history) > 1 else None

        # Create one-hot of intents to represent the current user action
        user_act_rep = np.zeros((self.num_intents,))
        user_act_rep[self.intents_dict[user_action[const.INTENT]]] = 1.0

        # Create bag of inform slots representation to represent the current user action
        user_inform_slots_rep = np.zeros((self.num_slots,))
        for key in user_action[const.INFORM_SLOTS].keys():
            user_inform_slots_rep[self.slots_dict[key]] = 1.0

        # Create bag of request slots representation to represent the current user action
        user_request_slots_rep = np.zeros((self.num_slots,))
        for key in user_action[const.REQUEST_SLOTS].keys():
            user_request_slots_rep[self.slots_dict[key]] = 1.0

        # Create bag of filled_in slots based on the current_slots
        current_slots_rep = np.zeros((self.num_slots,))
        for key in self.current_informs:
            current_slots_rep[self.slots_dict[key]] = 1.0

        # Encode last agent intent
        agent_act_rep = np.zeros((self.num_intents,))
        if last_agent_action:
            agent_act_rep[self.intents_dict[last_agent_action[const.INTENT]]] = 1.0

        # Encode last agent inform slots
        agent_inform_slots_rep = np.zeros((self.num_slots,))
        if last_agent_action:
            for key in last_agent_action[const.INFORM_SLOTS].keys():
                agent_inform_slots_rep[self.slots_dict[key]] = 1.0

        # Encode last agent request slots
        agent_request_slots_rep = np.zeros((self.num_slots,))
        if last_agent_action:
            for key in last_agent_action[const.REQUEST_SLOTS].keys():
                agent_request_slots_rep[self.slots_dict[key]] = 1.0

        # Value representation of the round num
        turn_rep = np.zeros((1,)) + self.round_num / 5.

        # One-hot representation of the round num
        turn_onehot_rep = np.zeros((self.max_round_num,))
        turn_onehot_rep[self.round_num - 1] = 1.0

        # Representation of DB query results (scaled counts)
        kb_count_rep = np.zeros((self.num_slots + 1,)) + db_results_dict[const.KB_MATCHING_ALL_CONSTRAINTS] / 100.
        for key in db_results_dict.keys():
            if key in self.slots_dict:
                kb_count_rep[self.slots_dict[key]] = db_results_dict[key] / 100.

        # Representation of DB query results (binary)
        kb_binary_rep = np.zeros((self.num_slots + 1,)) + \
            np.sum(db_results_dict[const.KB_MATCHING_ALL_CONSTRAINTS] > 0.)
        for key in db_results_dict.keys():
            if key in self.slots_dict:
                kb_binary_rep[self.slots_dict[key]] = np.sum(db_results_dict[key] > 0.)

        state_representation = np.hstack(
            [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
             agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep,
             kb_count_rep]).flatten()

        return state_representation

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
            inform_slots = self.db_helper.fill_inform_slot(agent_action[const.INFORM_SLOTS], self.current_informs)
            agent_action[const.INFORM_SLOTS] = inform_slots
            assert agent_action[const.INFORM_SLOTS]
            for key, value in list(agent_action[const.INFORM_SLOTS].items()):
                assert key != const.MATCH_FOUND
                assert value != const.PLACEHOLDER, 'KEY: {}'.format(key)
                self.current_informs[key] = value
        # If intent is match_found then fill the action informs with the matches informs (if there is a match)
        elif agent_action[const.INTENT] == const.MATCH_FOUND:
            assert not agent_action[const.INFORM_SLOTS], 'Cannot inform and have intent of match found!'
            db_results = self.db_helper.get_db_results(self.current_informs)
            if db_results:
                # Arbitrarily pick the first value of the dict
                key, value = list(db_results.items())[0]
                agent_action[const.INFORM_SLOTS] = copy.deepcopy(value)
                agent_action[const.INFORM_SLOTS][self.match_key] = str(key)
            else:
                agent_action[const.INFORM_SLOTS][self.match_key] = const.NO_MATCH
            self.current_informs[self.match_key] = agent_action[const.INFORM_SLOTS][self.match_key]
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

        for key, value in user_action[const.INFORM_SLOTS].items():
            self.current_informs[key] = value
        user_action.update({const.ROUND: self.round_num + 1, const.SPEAKER_TYPE: const.USR_SPEAKER_VAL})
        self.history.append(user_action)
        self.round_num += 1
