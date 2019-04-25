from dialogue_system.users import RealUser, RuleBasedUserSimulator
from dialogue_system.users.error_model_controller import ErrorModelController
from dialogue_system.dm.agents import DQNAgent
from dialogue_system.dm.dst import StateTracker
from dialogue_system.utils.util import remove_empty_slots

import pickle
import json
import math


class DialogueSystem:

    def __init__(self, params):
        # Load file path params
        file_path_dict = params['db_file_paths']
        self.database_path = file_path_dict['database']
        self.dict_path = file_path_dict['dict']
        self.goals_path = file_path_dict['user_goals']

        # Load run params
        run_dict = params['run']
        self.use_simulator = run_dict['usersim']
        self.warmup_mem = run_dict['warmup_mem']
        self.num_ep_run = run_dict['num_ep_run']
        self.train_freq = run_dict['train_freq']
        self.max_round_num = run_dict['max_round_num']
        self.success_rate_threshold = run_dict['success_rate_threshold']

        # Load movie DB
        # Note: If you get an unpickling error here then run 'pickle_converter.py' and it should fix it
        self.database = pickle.load(open(self.database_path, 'rb'), encoding='latin1')

        # Clean DB
        remove_empty_slots(self.database)

        # Load movie dict
        self.db_dict = pickle.load(open(self.dict_path, 'rb'), encoding='latin1')

        # Load goal File
        self.user_goals = pickle.load(open(self.goals_path, 'rb'), encoding='latin1')

        # Init. Objects
        if self.use_simulator:
            self.user = RuleBasedUserSimulator(self.user_goals, params, self.database)
        else:
            self.user = RealUser(params)
        self.emc = ErrorModelController(self.db_dict, params)
        self.state_tracker = StateTracker(self.database, params)
        self.dqn_agent = DQNAgent(self.state_tracker.get_state_size(), params)

        # The metrics to store
        self.performance_path = params['agent']['performance_path']
        self.performance_metrics = {}
        self.performance_metrics['success_rate'] = {}
        self.performance_metrics['avg_round'] = {}
        self.performance_metrics['avg_reward'] = {}

    def save_performance_records(self, records):
        """Save performance numbers."""
        try:
            json.dump(records, open(self.performance_path, "w"), indent=2)
            print(f'saved model in {self.performance_path}')
        except Exception as e:
            print(f'Error: Writing model fails: {self.performance_path}')
            print(e)

    def run_round(self, state, warmup=False, train=True):
        # 1) Agent takes action given state tracker's representation of dialogue (state)
        agent_action_index, agent_action = self.dqn_agent.get_action(state, use_rule=warmup)
        # 2) Update state tracker with the agent's action
        self.state_tracker.update_state_agent(agent_action)
        # print(f'Agent Action: {agent_action}')
        # 3) User takes action given agent action
        user_action, reward, done, success = self.user.step(agent_action)
        if not done:
            # 4) Infuse error into semantic frame level of user action
            self.emc.infuse_error(user_action)
        # 5) Update state tracker with user action
        self.state_tracker.update_state_user(user_action)
        # print(f'User Action: {user_action}\n')
        # 6) Get next state and add experience
        next_state = self.state_tracker.get_state(done)
        if train or warmup:
            self.dqn_agent.add_experience(state, agent_action_index, reward, next_state, done)

        return next_state, reward, done, success

    def warmup_run(self):
        """
        Runs the warmup stage of training which is used to fill the agents memory.

        The agent uses it's rule-based policy to make actions. The agent's memory is filled as this runs.
        Loop terminates when the size of the memory is equal to WARMUP_MEM or when the memory buffer is full.

        """

        print('Warmup Started...')
        total_step = 0
        while total_step != self.warmup_mem and not self.dqn_agent.is_memory_full():
            # Reset episode
            self.episode_reset()
            done = False
            # Get initial state from state tracker
            state = self.state_tracker.get_state()
            while not done:
                next_state, _, done, _ = self.run_round(state, warmup=True)
                total_step += 1
                state = next_state

        print('...Warmup Ended')

    def episode_reset(self):
        """
        Resets the episode/conversation.

        Called in warmup and train to reset the state tracker, user and agent. Also get's the initial user action.
        """

        # First reset the state tracker
        self.state_tracker.reset()
        # Then pick an init user action
        user_action = self.user.reset()
        # Infuse with error
        self.emc.infuse_error(user_action)
        # And update state tracker
        self.state_tracker.update_state_user(user_action)
        # Finally, reset agent
        self.dqn_agent.reset()

    def train_run(self):
        """
        Runs the loop that trains the agent.

        Trains the agent on the goal-oriented chatbot task. Training of the agent's neural network occurs
        every episode that TRAIN_FREQ is a multiple of. Terminates when the episode reaches NUM_EP_TRAIN.
        """

        print('Training Started...')
        episode = 0
        period_reward_total = 0
        period_success_total = 0
        period_round_total = 0
        success_rate_best = 0.0
        while episode < self.num_ep_run:
            self.episode_reset()
            episode += 1
            done = False
            state = self.state_tracker.get_state()
            rounds = 0
            while not done:
                next_state, reward, done, success = self.run_round(state)
                period_reward_total += reward
                state = next_state
                rounds += 1

            period_success_total += success
            period_round_total += rounds

            # Train
            if episode % self.train_freq == 0:
                # evaluate metrics
                self.performance_metrics['success_rate'][episode] = period_success_total / self.train_freq
                self.performance_metrics['avg_reward'][episode] = period_reward_total / self.train_freq
                self.performance_metrics['avg_round'][episode] = period_round_total / self.train_freq

                # Check success rate
                success_rate = period_success_total / self.train_freq
                avg_reward = period_reward_total / self.train_freq
                # Flush
                if success_rate >= success_rate_best and success_rate >= self.success_rate_threshold:
                    self.dqn_agent.empty_memory()
                # Update current best success rate
                if success_rate > success_rate_best:
                    print(f'Episode: {episode} NEW BEST SUCCESS RATE: {success_rate} Avg Reward: {avg_reward}')
                    success_rate_best = success_rate
                    self.dqn_agent.save_weights()
                period_success_total = 0
                period_reward_total = 0
                period_round_total = 0
                # Copy
                self.dqn_agent.copy()
                # Train
                self.dqn_agent.train()
        print('...Training Ended')
        self.save_performance_records(self.performance_metrics)

    def train(self):
        self.warmup_run()
        self.train_run()

    def test(self):
        """
        Runs the loop that tests the agent.

        Tests the agent on the goal-oriented chatbot task. Only for evaluating a trained agent.
        Terminates when the episode reaches NUM_EP_TEST.

        """

        print('Testing Started...')
        episode = 0
        while episode < self.num_ep_run:
            self.episode_reset()
            episode += 1
            ep_reward = 0
            done = False
            # Get initial state from state tracker
            state = self.state_tracker.get_state()
            success = False
            while not done:
                next_state, reward, done, success = self.run_round(state, train=False)
                ep_reward += reward
                state = next_state
            print(f'Episode: {episode} Success: {success} Reward: {ep_reward}')
        print('...Testing Ended')
