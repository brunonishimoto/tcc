from slackclient import SlackClient
from utils.util import log

import dialogue_system.dialogue_config as cfg
import dialogue_system.constants as const
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
        self.max_round = config['run']['max_round_num']

        # instantiate Slack client
        self.slack_client = SlackClient('################TOKEN###################')

        # starterbot's user ID in Slack: value is assigned after the bot starts up
        self.starterbot_id = None

        # constants
        self.RTM_READ_DELAY = 1 # 1 second delay between reading from RTM
        self.MENTION_REGEX = "^<@(|[WU].+?)>(.*)"

        if self.slack_client.rtm_connect(with_team_state=False):
            log(['debug'], "Connected to the Slack!")
            # Read bot's user ID by calling Web API method `auth.test`
            self.starterbot_id = self.slack_client.api_call("auth.test")["user_id"]
        else:
            log(['debug'], "Connection failed")
            exit()

    def parse_events(self, slack_events):
        """
            Parses a list of events coming from the Slack RTM API to find bot commands.
            If a bot command is found, this function returns a tuple of command and channel.
            If its not found, then this function returns None, None.
        """
        for event in slack_events:
            if event["type"] == "message" and not "subtype" in event:
                user_id, message = self.parse_direct_mention(event["text"])
                if user_id == self.starterbot_id:
                    return message, event["channel"]
                if event["channel"] == "DJTA4UUSY":
                    return event["text"], event["channel"]
        return None, None

    def parse_direct_mention(self, message_text):
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

        return self.__return_response()

    def __return_response(self):
        """
        Asks user in console for response then receives a response as input.
        """
        command = None
        while not command:
            command, self.channel = self.parse_events(self.slack_client.rtm_read())
            time.sleep(self.RTM_READ_DELAY)

        return {'nl': command}

    def __return_success(self):
        """
        Ask the user in console to input success (-1, 0 or 1) for (loss, neither loss nor win, win).

        Returns:
            int: Success: -1, 0 or 1
        """

        success = -2
        while success not in (-1, 0, 1):
            command, self.channel = self.parse_events(self.slack_client.rtm_read())
            success = int(command)
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

        log(['debug'], f'Agent Action: {agent_action}')

        # Sends the response back to the channel
        self.slack_client.api_call(
            "chat.postMessage",
            channel=self.channel,
            text=agent_action['nl']
        )

        done = False
        user_response = {const.INTENT: '', const.REQUEST_SLOTS: {}, const.INFORM_SLOTS: {}}

        # First check round num, if equal to max then fail
        if agent_action[const.ROUND] == self.max_round:
            success = const.FAILED_DIALOG
            user_response[const.INTENT] = const.CLOSING
        else:
            user_response = self.__return_response()
            success = self.__return_success()

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
