from slackclient import SlackClient
from utils.util import log

import dialogue_system.dialogue_config as cfg
import dialogue_system.constants as const
import json
import pickle
import random
import time
import re


class RealUserSlack():
    """Connects a real user to the conversation through the console."""

    def __init__(self, config):
        """
        The constructor for User.

        Parameters:
            config (dict): Loaded config as dict
        """

        goals_path = config['db_file_paths']['user_goals']
        self.goal_list = pickle.load(open(goals_path, 'rb'), encoding='latin1')
        self.goal_list = [
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'birmingham', 'numberofpeople': '1', 'theater': 'carmike summit 16', 'state': 'al', 'starttime': 'around 2pm', 'date': 'today', 'moviename': 'zootopia'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'amc pacific place 11 theater', 'starttime': '9:00 pm', 'date': 'tomorrow', 'moviename': 'deadpool'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'birmingham', 'numberofpeople': '4', 'theater': 'carmike summit 16', 'state': 'al', 'starttime': 'around 6pm', 'date': 'today', 'moviename': 'deadpool'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'regal meridian 16', 'starttime': '9:10 pm', 'date': 'tomorrow', 'moviename': 'zootopia'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'regal meridian 16', 'starttime': '9:30 pm', 'date': 'tomorrow', 'moviename': 'the witch'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'regal meridian 16', 'starttime': '9:30 pm', 'date': 'tomorrow', 'moviename': 'the witch'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '6', 'theater': 'amc lowes oak tree', 'starttime': '7:10 pm', 'date': 'tomorrow', 'moviename': 'triple 9'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'regal meridian 16', 'starttime': '9:10 pm', 'date': 'tomorrow', 'moviename': 'zootopia'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'regal meridian 16', 'starttime': '8:45 pm', 'date': 'tomorrow', 'moviename': 'the big short'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'regal meridian 16', 'starttime': '8:45 pm', 'date': 'tomorrow', 'moviename': 'the big short'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'regal meridian 16', 'starttime': '9:10 pm', 'date': 'tomorrow', 'moviename': 'zootopia'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'amc lowes oak tree 6', 'starttime': '4:50 pm', 'date': 'tomorrow', 'moviename': 'race'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'birmingham', 'numberofpeople': '3', 'theater': 'carmike summit 16', 'state': 'al', 'starttime': '2pm', 'date': 'saturday', 'moviename': 'deadpool'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'regal meridian 16', 'starttime': '9:10 pm', 'date': 'tomorrow', 'moviename': 'zootopia'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'regal meridian 16', 'starttime': '9:10 pm', 'date': 'tomorrow', 'moviename': 'zootopia'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'birmingham', 'numberofpeople': '2', 'theater': 'carmike summit 16', 'state': 'al', 'starttime': 'around 2pm', 'date': 'today', 'moviename': 'london had fallen'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'birmingham', 'numberofpeople': '2', 'theater': 'carmike summit 16', 'state': 'al', 'starttime': 'around 5pm', 'date': 'tomorrow', 'moviename': 'deadpool'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'regal meridian 16', 'starttime': '9:00 pm', 'date': 'tomorrow', 'moviename': 'spotlight'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'regal meridian 16', 'starttime': '9:00 pm', 'date': 'tomorrow', 'moviename': 'spotlight'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'birmingham', 'numberofpeople': '2', 'theater': 'carmike 16', 'state': 'al', 'starttime': '2pm', 'date': 'today', 'moviename': 'zootopia'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'regal meridian 16', 'starttime': '9:00 pm', 'date': 'tomorrow', 'moviename': 'spotlight'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'regal meridian 16', 'starttime': '9:10 pm', 'date': 'tomorrow', 'moviename': 'zootopia'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'amc pacific place 11 theater', 'starttime': '9:00 pm', 'date': 'tomorrow', 'moviename': 'deadpool'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'regal meridian 16', 'starttime': '9:10 pm', 'date': 'tomorrow', 'moviename': 'zootopia'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'birmingham', 'numberofpeople': '2', 'theater': 'carmike summit 16', 'state': 'al', 'starttime': 'around 2pm', 'date': 'sunday', 'moviename': 'deadpool'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'regal meridian 16', 'starttime': '8:45 pm', 'date': 'tomorrow', 'moviename': 'big short'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'regal meridian 16', 'starttime': '9:30 pm', 'date': 'tomorrow', 'moviename': 'the witch'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'regal meridian 16', 'starttime': '9:10 pm', 'date': 'tomorrow', 'moviename': 'zootopia'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'regal meridian 16', 'starttime': '9:00 pm', 'date': 'tomorrow', 'moviename': 'spotlight'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'regal meridian 16', 'starttime': '9:30 pm', 'date': 'tomorrow', 'moviename': 'the witch'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'regal meridian 16', 'starttime': '9:30 pm', 'date': 'tomorrow', 'moviename': 'the witch'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'birmingham', 'numberofpeople': '1', 'theater': 'carmike summit 16', 'state': 'al', 'starttime': '2pm', 'date': 'tomorrow', 'moviename': 'deadpool'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'birmingham', 'numberofpeople': '2', 'theater': 'carmike summit 16', 'state': 'al', 'starttime': 'around 2pm', 'date': 'thursday', 'moviename': 'deadpool'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '6', 'theater': 'amc lowes oak tree', 'starttime': '7:10 pm', 'date': 'tomorrow', 'moviename': 'triple 9'}},
            {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'amc pacific place 11 theater', 'starttime': '9:00 pm', 'date': 'tomorrow', 'moviename': 'deadpool'}}
        ]

        # Flag if we give a goal to the user
        self.get_goal = config['user']['give_goal']

        self.max_round = config['run']['max_round_num']

        # instantiate Slack client
        self.slack_client = SlackClient('xoxb-628422054787-639344980276-LsfpqMrTJMNhsisl2fIDMpxH')

        # starterbot's user ID in Slack: value is assigned after the bot starts up
        self.starterbot_id = None

        # constants
        self.RTM_READ_DELAY = 1 # 1 second delay between reading from RTM
        self.MENTION_REGEX = "^<@(|[WU].+?)>(.*)"

        if self.slack_client.rtm_connect(with_team_state=False):
            log(['debug'], "Connected to the Slack!")
            # Read bot's user ID by calling Web API method `auth.test`
            self.starterbot_id = self.slack_client.api_call("auth.test")["user_id"]

            # self.channel = "CJVLHDT55" #geral
            self.channel = "CRS3F74NS" #go-bot
            # self.channel = "DJTA4UUSY"
        else:
            log(['debug'], "Connection failed")
            exit()

    def __get_channel(self, channel):
        channels_call = self.slack_client.api_call("channels.list")
        channels_list = None

        if channels_call['ok']:
            channels_list = channels_call['channels']

        for c in channels_list:
            if c['name'] == channel:
                return c['id']

        raise Exception(f"No such channel: {channel}")

    def __parse_events(self, slack_events):
        """
            Parses a list of events coming from the Slack RTM API to find bot commands.
            If a bot command is found, this function returns a tuple of command and channel.
            If its not found, then this function returns None, None.
        """
        for event in slack_events:
            if event["type"] == "message" and not "subtype" in event:
                return event["text"]
                user_id, message = self.__parse_direct_mention(event["text"])
                if user_id == self.starterbot_id:
                    return message
                return event["text"]
                user_id, message = self.__parse_direct_mention(event["text"])
                if user_id == self.starterbot_id:
                    return message, event["channel"]
                if event["channel"] == self.channel:
                    return event["text"], event["channel"]
        return None

    def __parse_direct_mention(self, message_text):
        """
            Finds a direct mention (a mention that is at the beginning) in message text
            and returns the user ID which was mentioned. If there is no direct mention, returns None
        """
        matches = re.search(self.MENTION_REGEX, message_text)
        # the first group contains the username, the second group contains the remaining mess
        return (matches.group(1), matches.group(2).strip()) if matches else (None, None)

    def reset(self, episode=0, train=True):
        """
        Reset the user.

        Returns:
            dict: The user response
        """
        self.__wait_start_command()

        if self.get_goal:
            # sample_goal = random.choice(self.goal_list)
            sample_goal = {'request_slots': {}, 'diaact': 'request', 'inform_slots': {'city': 'seattle', 'numberofpeople': '2', 'theater': 'amc pacific place 11 theater', 'starttime': '9:00 pm', 'date': 'tomorrow', 'moviename': 'deadpool'}}
            log(['dialogue'], f"Goal: {sample_goal}")
            self.slack_client.api_call(
                "chat.postMessage",
                channel=self.channel,
                text=f"Your goal is:\n {json.dumps(sample_goal, indent=2)}"
            )

        return self.__return_response()

    def __return_response(self):
        """
        Asks user in console for response then receives a response as input.
        """
        command = None
        while not command:
            command = self.__parse_events(self.slack_client.rtm_read())
            time.sleep(self.RTM_READ_DELAY)

        return {'nl': command}

    def __return_success(self):
        """
        Ask the user in console to input success (-1, 0 or 1) for (loss, neither loss nor win, win).

        Returns:
            int: Success: -1, 0 or 1
        """

        success = -2
        self.slack_client.api_call(
            "chat.postMessage",
            channel=self.channel,
            text=f"The agent could complete the task? -- (0 or 1) for (no, yes)"
        )
        while success not in (0, 1):

            command = self.__parse_events(self.slack_client.rtm_read())
            if command:
                success = int(command)
        return success

    def __wait_start_command(self):
        command = None

        self.slack_client.api_call(
            "chat.postMessage",
            channel=self.channel,
            text=f"Type 'start' to init a conversation"
        )
        while command != 'start':
            command = self.__parse_events(self.slack_client.rtm_read())
            if command:
                command.lower()
            time.sleep(self.RTM_READ_DELAY)

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

        # log(['debug'], f'Agent Action: {agent_action}')

        done = False
        success = const.NO_OUTCOME_YET
        user_response = {const.INTENT: '', const.REQUEST_SLOTS: {}, const.INFORM_SLOTS: {}}

        # First check round num, if equal to max then fail
        if agent_action[const.ROUND] == self.max_round:
            success = const.FAILED_DIALOG
            self.slack_client.api_call(
                "chat.postMessage",
                channel=self.channel,
                text="I'm sorry. I could not complete the dialog in the maximum number of rounds. ::confused:"
            )
        else:
            # Sends the response back to the channel
            self.slack_client.api_call(
                "chat.postMessage",
                channel=self.channel,
                text=agent_action['nl']
            )

            if agent_action[const.INTENT] == const.CLOSING or agent_action[const.INTENT] == const.THANKS:
                user_response = self.__return_response()
                # success = self.__return_success()
                done = True
            else:
                user_response = self.__return_response()

            # log(['debug'], f"User: {user_response} ---- sucess: {success}")

        # if success == const.FAILED_DIALOG or success == const.SUCCESS_DIALOG:
        #     done = True

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
