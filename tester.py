import argparse
import json
from dialogue_system import DialogueSystem
from recorder import Recorder
import os


class Tester:

    def __init__(self, params):

        # Load run params
        run_dict = params['run']
        self.use_simulator = run_dict['usersim']
        self.warmup_mem = run_dict['warmup_mem']
        self.num_ep_run = run_dict['num_ep_run']
        self.num_ep_test = run_dict['num_ep_test']
        self.train_freq = run_dict['train_freq']
        self.max_round_num = run_dict['max_round_num']
        self.success_rate_threshold = run_dict['success_rate_threshold']
        self.sigma_init = params['agent']['sigma_init']
        self.sigma_stop = params['agent']['sigma_stop']
        self.sigma_decay = params['agent']['sigma_decay']

        self.dialogue_system = DialogueSystem(params)
        self.recorder = Recorder(params)

    def test(self, train_episode=0):
        """
        Runs the loop that tests the agent.

        Tests the agent on the goal-oriented chatbot task. Only for evaluating a trained agent.
        Terminates when the episode reaches NUM_EP_TEST.

        """

        # print('Testing Started...')
        episode = 0
        period_reward_total = 0
        period_success_total = 0
        period_round_total = 0
        while episode < self.num_ep_test:
            self.dialogue_system.reset(train=False, test_episode=episode)
            episode += 1
            ep_reward = 0
            done = False
            # Get initial state from state tracker
            state = self.dialogue_system.state_tracker.get_state()
            success = False
            rounds = 0
            while not done:
                next_state, reward, done, success = self.dialogue_system.run_round(state, train=False)
                ep_reward += reward
                state = next_state
                rounds += 1

            period_success_total += success
            period_round_total += rounds
            period_reward_total += ep_reward
            # print(f'Episode: {episode} Success: {success} Reward: {ep_reward}')
        self.performance_metrics['test']['success_rate'][train_episode] = period_success_total / self.num_ep_test
        self.performance_metrics['test']['avg_reward'][train_episode] = period_reward_total / self.num_ep_test
        self.performance_metrics['test']['avg_round'][train_episode] = period_round_total / self.num_ep_test
        # print('...Testing Ended')
        # self.save_performance_records()
