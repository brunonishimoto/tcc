import random
from dialogue_config import usersim_intents
import constants as const


class ErrorModelController:
    """Adds error to the user action."""

    def __init__(self, db_dict, params):
        """
        The constructor for ErrorModelController.

        Saves items in params, etc.

        Parameters:
            db_dict (dict): The database dict with format dict(string: list) where each key is the slot name and
                            the list is of possible values
            params (dict): Loaded params in dict
        """

        self.movie_dict = db_dict
        self.slot_error_prob = params['emc']['slot_error_prob']
        self.slot_error_mode = params['emc']['slot_error_mode']  # [0, 3]
        self.intent_error_prob = params['emc']['intent_error_prob']
        self.intents = usersim_intents

    def infuse_error(self, frame):
        """
        Takes a semantic frame/action as a dict and adds 'error'.

        Given a dict/frame it adds error based on specifications in params. It can either replace slot values,
        replace slot and its values, delete a slot or do all three. It can also randomize the intent.

        Parameters:
            frame (dict): format dicconst.INTENTntent': '', 'inform_slots': {}, 'request_slots': {}, 'round': int,
                          'speaker': 'User')
        """

        informs_dict = frame[const.INFORM_SLOTS]
        for key in list(frame[const.INFORM_SLOTS].keys()):
            assert key in self.movie_dict
            if random.random() < self.slot_error_prob:
                if self.slot_error_mode == 0:  # replace the slot_value only
                    self._slot_value_noise(key, informs_dict)
                elif self.slot_error_mode == 1:  # replace slot and its values
                    self._slot_noise(key, informs_dict)
                elif self.slot_error_mode == 2:  # delete the slot
                    self._slot_remove(key, informs_dict)
                else:  # Combine all three
                    rand_choice = random.random()
                    if rand_choice <= 0.33:
                        self._slot_value_noise(key, informs_dict)
                    elif rand_choice > 0.33 and rand_choice <= 0.66:
                        self._slot_noise(key, informs_dict)
                    else:
                        self._slot_remove(key, informs_dict)
        if random.random() < self.intent_error_prob:  # add noise for intent level
            frame[const.INTENT] = random.choice(self.intents)

    def _slot_value_noise(self, key, informs_dict):
        """
        Selects a new value for the slot given a key and the dict to change.

        Parameters:
            key (string)
            informs_dict (dict)
        """

        informs_dict[key] = random.choice(self.movie_dict[key])

    def _slot_noise(self, key, informs_dict):
        """
        Replaces current slot given a key in the informs dict with a new slot and selects a random value for
        this new slot.

        Parameters:
            key (string)
            informs_dict (dict)
        """

        informs_dict.pop(key)
        random_slot = random.choice(list(self.movie_dict.keys()))
        informs_dict[random_slot] = random.choice(self.movie_dict[random_slot])

    def _slot_remove(self, key, informs_dict):
        """
        Removes the slot given the key from the informs dict.

        Parameters:
            key (string)
            informs_dict (dict)
        """

        informs_dict.pop(key)
