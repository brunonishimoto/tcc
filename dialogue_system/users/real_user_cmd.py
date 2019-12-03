import dialogue_system.dialogue_config as cfg
import dialogue_system.constants as const

import pickle
import random
import json


class RealUserCMD():
    """Connects a real user to the conversation through the console."""

    def __init__(self, config):
        """
        The constructor for User.


        Parameters:
            config (dict): Loaded config as dict
        """

        goals_path = config['db_file_paths']['user_goals']
        self.goal_list = pickle.load(open(goals_path, 'rb'), encoding='latin1')

        # Flag if we give a goal to the user
        self.get_goal = config['user']['give_goal']

        self.max_round = config['run']['max_round_num']

    def reset(self, episode=0, train=True):
        """
        Reset the user.

        Returns:
            dict: The user response
        """
        self.__wait_start_command()


        if self.get_goal:
            sample_goal = random.choice(self.goal_list)

            print(f"Your goal is:\n {json.dumps(sample_goal, indent=2)}")

        return self.__return_response()

    def __return_response(self):
        """
        Asks user in console for response then receives a response as input.
        """

        input_string = input('Response: ')

        return {'nl': input_string}

    def __return_success(self):
        """
        Ask the user in console to input success (-1, 0 or 1) for (loss, neither loss nor win, win).

        Returns:
            int: Success: -1, 0 or 1
        """

        success = -2
        while success not in (-1, 0, 1):
            success = int(input(f"The agent could complete the task? -- (-1, 0 or 1) for (loss, neither loss nor win, win)\nAnswer: "))
        return success

    def __wait_start_command(self):
        command = None

        while command != 'start':
            command = input("Type 'start' to init a conversation: ").lower()


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

        print(f'Agent Action: {agent_action}')

        done = False
        success = const.NO_OUTCOME_YET
        user_response = {const.INTENT: '', const.REQUEST_SLOTS: {}, const.INFORM_SLOTS: {}}

        # First check round num, if equal to max then fail
        # if agent_action[const.ROUND] == self.max_round:
        #     success = const.FAILED_DIALOG
        #     user_response[const.INTENT] = const.THANKS
        # else:
        if agent_action[const.INTENT] == const.THANKS:
            user_response = self.__return_response()
            success = self.__return_success()
        else:
            user_response = self.__return_response()

        if success == const.FAILED_DIALOG or success == const.SUCCESS_DIALOG:
            done = True

        reward = self.__reward_function(success)

        return user_response, reward, done, True if success is 1 else False

    def __reward_function(self, success):
        """
        Return the reward given the success.

        Return -1 + -max_round if success is FAIL, -1 + 2 * max_round if success is SUCCESS and -1 otherwise.

        Parameters:
            success (int)

        Returns:
            int: Reward
        """

        reward = -1
        if success == const.FAILED_DIALOG:
            reward += - self.max_round
        elif success == const.SUCCESS_DIALOG:
            reward += 2 * self.max_round
        return reward
