import pickle
import json
import math
import random
import collections

import dialogue_system.users as users
import dialogue_system.dm.agents as agents
import dialogue_system.dm.dst as state_trackers
import dialogue_system.nlu as nlus
import dialogue_system.nlg as nlgs

from dialogue_system.users.error_model_controller import ErrorModelController
from utils.util import remove_empty_slots, log


class DialogueSystem:

    def __init__(self, config):

        # Init. the components of the dialogue system
        self.user = users.load(config)
        self.emc = ErrorModelController(config)
        self.nlu = nlus.load(config)
        self.nlg = nlgs.load(config)
        self.state_tracker = state_trackers.load(config)
        self.agent = agents.load(config)
        self.agent.build_models(self.state_tracker.get_state_size())

        self.use_nl = config['use_nl']
        self.real_user = config['real_user']
        self.recurrent = config['recurrent']

        if self.recurrent:
            self.episode_experience = []
        self.state = None

    def run_round(self, step=None, use_rule=False, train=True):
        # 1) Agent takes action given state tracker's representation of dialogue (state)
        agent_action_index, agent_action = self.agent.get_action(self.state, step=step, use_rule=use_rule, train=train)

        # 2) Update state tracker with the agent's action
        self.state_tracker.update_state_agent(agent_action)
        if self.use_nl:
            agent_action['nl'] = self.nlg.convert_diaact_to_nl(agent_action, 'agt')
        agent_action = self.__transform_action(agent_action)
        log(['dialogue'], f'Agent: {agent_action}')

        # 3) User takes action given agent action
        user_action, reward, done, success = self.user.step(agent_action)

        if not done:
            # 4) Infuse error into semantic frame level of user action
            if self.use_nl and not self.real_user:
                user_action['nl'] = self.nlg.convert_diaact_to_nl(agent_action, 'usr')
            user_action = self.__transform_action(user_action, infuse_error=True)
            log(['debug', 'dialogue'], f'User: {user_action}')

        # 5) Update state tracker with user action
        self.state_tracker.update_state_user(user_action)

        # 6) Get next state and add experience
        next_state = self.state_tracker.get_state(done)

        if train:
            if self.recurrent:
                self.episode_experience.append((self.state, agent_action_index, reward, next_state, done))
                if done:
                    self.agent.add_experience(self.episode_experience)
                    self.episode_experience = []
            else:
                self.agent.add_experience(self.state, agent_action_index, reward, next_state, done)

        # Update the dialogue state
        self.state = next_state
        return self.state, reward, done, success

    def reset(self, episode, train=True):
        """
        Resets the episode/conversation.

        Called in warmup and train to reset the state tracker, user and agent. Also get's the initial user action.
        """

        # First reset the state tracker
        self.state_tracker.reset()
        # Then pick an init user action
        user_action = self.user.reset(episode, train)
        if self.use_nl and not self.real_user:
            user_action['nl'] = self.nlg.convert_diaact_to_nl(user_action, 'usr')
        # if nl transform in frame, if frame use emc
        user_action = self.__transform_action(user_action, infuse_error=True)
        log(['dialogue'], f'User: {user_action}')
        # And update state tracker
        self.state_tracker.update_state_user(user_action)
        self.state = self.state_tracker.get_state()
        # Finally, reset agent
        self.episode_experience = []
        self.agent.reset()

    # TODO: think in a better name for this function
    def __transform_action(self, action, infuse_error=False):
        if self.use_nl:
            action.update(self.nlu.generate_dia_act(action['nl']))
        elif infuse_error:
            action = self.emc.infuse_error(action)
        return action
