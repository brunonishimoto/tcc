import dialogue_system.dialogue_config as cfg
import dialogue_system.constants as const
import numpy as np
import random
import pickle

from utils.util import log

class ErrorModelController:
    """Adds error to the user action."""

    def __init__(self, config):
        """
        The constructor for ErrorModelController.

        Saves items in config, etc.

        Parameters:
            config (dict): Loaded config in dict
        """

        # Load movie dict
        dict_path = config['db_file_paths']['dict']
        db_dict = pickle.load(open(dict_path, 'rb'), encoding='latin1')

        self.movie_dict = db_dict
        self.slot_error_prob = config['emc']['slot_error_prob']
        self.slot_error_mode = config['emc']['slot_error_mode']  # [0, 3]
        self.intent_error_prob = config['emc']['intent_error_prob']
        self.intents = cfg.usersim_intents

    def infuse_error(self, frame):
        """
        Takes a semantic frame/action as a dict and adds 'error'.

        Given a dict/frame it adds error based on specifications in config. It can either replace slot values,
        replace slot and its values, delete a slot or do all three. It can also randomize the intent.

        Parameters:
            frame (dict): format dicconst.INTENTntent': '', 'inform_slots': {}, 'request_slots': {}, 'round': int,
                          'speaker': 'User')
        """
        cfg.correct = frame
        if self.slot_error_prob > 0:
            informs_dict = frame[const.INFORM_SLOTS]
            for key in list(frame[const.INFORM_SLOTS].keys()):
                if key == cfg.usersim_default_key:
                    continue
                assert key in self.movie_dict
                if random.random() < self.slot_error_prob:
                    if self.slot_error_mode == 0:  # replace the slot_value only
                        self.__slot_value_noise(key, informs_dict)
                    elif self.slot_error_mode == 1:  # replace slot and its values
                        self.__slot_noise(key, informs_dict)
                    elif self.slot_error_mode == 2:  # delete the slot
                        self.__slot_remove(key, informs_dict)
                    else:  # Combine all three
                        choice = np.random.choice([0, 1, 2], p=[0.3, 0.3, 0.4])
                        if choice == 0:
                            self.__slot_value_noise(key, informs_dict)
                        elif choice == 1:
                            self.__slot_noise(key, informs_dict)
                        elif choice == 2:
                            self.__slot_remove(key, informs_dict)
        if random.random() < self.intent_error_prob:  # add noise for intent level
            frame[const.INTENT] = random.choice(self.intents)

        return frame

    def __slot_value_noise(self, key, informs_dict):
        """
        Selects a new value for the slot given a key and the dict to change.

        Parameters:
            key (string)
            informs_dict (dict)
        """

        informs_dict[key] = random.choice(self.movie_dict[key])

    def __slot_noise(self, key, informs_dict):
        """
        Replaces current slot given a key in the informs dict with a new slot and selects a random value for
        this new slot.

        Parameters:
            key (string)
            informs_dict (dict)
        """

        informs_dict.pop(key)
        random_slot = random.choice(list(cfg.all_slots))
        informs_dict[random_slot] = random.choice(self.movie_dict[random_slot])

    def __slot_remove(self, key, informs_dict):
        """
        Removes the slot given the key from the informs dict.

        Parameters:
            key (string)
            informs_dict (dict)
        """

        informs_dict.pop(key)
