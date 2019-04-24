from dialogue_system.dialogue_config import usersim_intents, all_slots
from dialogue_system.utils.util import reward_function
import dialogue_system.constants as const


class RealUser():
    """Connects a real user to the conversation through the console."""

    def __init__(self, params):
        """
        The constructor for User.

        Parameters:
            params (dict): Loaded params as dict
        """
        self.max_round = params['run']['max_round_num']

    def reset(self):
        """
        Reset the user.

        Returns:
            dict: The user response
        """

        return self._return_response()

    def _return_response(self):
        """
        Asks user in console for response then receives a response as input.

        Format must be like this: request/moviename: room, date: friday/starttime, city, theater
        or inform/moviename: zootopia/
        or request//starttime
        or done//
        intents, informs keys and values, and request keys and values cannot contain / , :

        Returns:
            dict: The response of the user
        """

        response = {const.INTENT: '', const.INFORM_SLOTS: {}, const.REQUEST_SLOTS: {}}
        while True:
            input_string = input('Response: ')
            chunks = input_string.split('/')

            intent_correct = True
            if chunks[0] not in usersim_intents:
                intent_correct = False
            response[const.INTENT] = chunks[0]

            informs_correct = True
            if len(chunks[1]) > 0:
                informs_items_list = chunks[1].split(', ')
                for inf in informs_items_list:
                    inf = inf.split(': ')
                    if inf[0] not in all_slots:
                        informs_correct = False
                        break
                    response[const.INFORM_SLOTS][inf[0]] = inf[1]

            requests_correct = True
            if len(chunks[2]) > 0:
                requests_key_list = chunks[2].split(', ')
                for req in requests_key_list:
                    if req not in all_slots:
                        requests_correct = False
                        break
                    response[const.REQUEST_SLOTS][req] = const.UNKNOWN

            if intent_correct and informs_correct and requests_correct:
                break

        return response

    def _return_success(self):
        """
        Ask the user in console to input success (-1, 0 or 1) for (loss, neither loss nor win, win).

        Returns:
            int: Success: -1, 0 or 1
        """

        success = -2
        while success not in (-1, 0, 1):
            success = int(input('Success?: '))
        return success

    def step(self, agent_action):
        """
        Return the user's response, reward, done and success.

        Parameters:
            agent_action (dict): The current action of the agent

        Returns:
            dict: User response
            int: Reward
            bool: Done flag
            int: Success: -1, 0 or 1 for loss, neither win nor loss, win
        """

        # Assertions ----
        # No unk in agent action informs
        for value in agent_action[const.INFORM_SLOTS].values():
            assert value != const.UNKNOWN
            assert value != const.PLACEHOLDER
        # No PLACEHOLDER in agent_action at all
        for value in agent_action[const.REQUEST_SLOTS].values():
            assert value != const.PLACEHOLDER
        # ---------------

        print('Agent Action: {}'.format(agent_action))

        done = False
        user_response = {const.INTENT: '', const.REQUEST_SLOTS: {}, const.INFORM_SLOTS: {}}

        # First check round num, if equal to max then fail
        if agent_action[const.ROUND] == self.max_round:
            success = const.FAILED_DIALOG
            user_response[const.INTENT] = const.DONE
        else:
            user_response = self._return_response()
            success = self._return_success()

        if success == const.FAILED_DIALOG or success == const.SUCCESS_DIALOG:
            done = True

        assert const.UNKNOWN not in user_response[const.INFORM_SLOTS].values()
        assert const.PLACEHOLDER not in user_response[const.REQUEST_SLOTS].values()

        reward = reward_function(success, self.max_round)

        return user_response, reward, done, True if success is 1 else False
