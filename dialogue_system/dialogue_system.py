from dialogue_system.users import RealUser, RuleBasedUserSimulator
from dialogue_system.users.error_model_controller import ErrorModelController
from dialogue_system.dm.agents import DQNEpsilonDecay
from dialogue_system.dm.dst import StateTracker
from dialogue_system.utils.util import remove_empty_slots

import pickle
import json
import math
import random
import collections


class DialogueSystem:

    def __init__(self, params):

        # Init. Objects
        # TODO: change the objects initialization with a load
        #       function in __init__ with globals()
        # if self.use_simulator:
        self.user = RuleBasedUserSimulator(params)
        # else:
        # self.user = RealUser(params)

        self.emc = ErrorModelController(params)
        self.state_tracker = StateTracker(params)
        self.dqn_agent = DQNEpsilonDecay(self.state_tracker.get_state_size(), params)

        self.state = None

    def run_round(self, use_rule=False, train=True):
        # 1) Agent takes action given state tracker's representation of dialogue (state)
        agent_action_index, agent_action = self.dqn_agent.get_action(self.state, use_rule=use_rule, train=train)
        # 2) Update state tracker with the agent's action
        self.state_tracker.update_state_agent(agent_action)
        # 3) User takes action given agent action
        user_action, reward, done, success = self.user.step(agent_action)
        if not done:
            # 4) Infuse error into semantic frame level of user action
            self.emc.infuse_error(user_action)
        # 5) Update state tracker with user action
        self.state_tracker.update_state_user(user_action)
        # 6) Get next state and add experience
        next_state = self.state_tracker.get_state(done)

        if train:
            self.dqn_agent.add_experience(self.state, agent_action_index, reward, next_state, done)

        # Update the dialogue state
        self.state = next_state

        return self.state, reward, done, success

    def get_state(self):
        return self.state

    def reset(self, train=True, test_episode=None):
        """
        Resets the episode/conversation.

        Called in warmup and train to reset the state tracker, user and agent. Also get's the initial user action.
        """

        # First reset the state tracker
        self.state_tracker.reset()
        # Then pick an init user action
        user_action = self.user.reset(train, test_episode)
        # Infuse with error
        self.emc.infuse_error(user_action)
        # And update state tracker
        self.state_tracker.update_state_user(user_action)
        self.state = self.state_tracker.get_state()
        # Finally, reset agent
        self.dqn_agent.reset()
