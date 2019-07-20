import argparse
import json
from dialogue_system import DialogueSystem
from recorder import Recorder
import os
import random


class Trainer:

    def __init__(self, params):

        # Load run params
        run_dict = params['run']
        self.use_simulator = run_dict['usersim']

        self.warmup_mem = run_dict['warmup_mem']
        self.num_ep_run = run_dict['num_ep_run']
        self.train_freq = run_dict['train_freq']

        self.max_round_num = run_dict['max_round_num']
        self.success_rate_threshold = run_dict['success_rate_threshold']

        # sigma parameter for using designer's knowledge
        self.sigma_init = params['agent']['sigma_init']
        self.sigma_stop = params['agent']['sigma_stop']
        self.sigma_decay = params['agent']['sigma_decay']

        self.dialogue_system = DialogueSystem(params)
        self.recorder = Recorder(params)

        self.performance_metrics = {}
        self.performance_metrics['train'] = {}
        self.performance_metrics['train']['success_rate'] = {}
        self.performance_metrics['train']['avg_reward'] = {}
        self.performance_metrics['train']['avg_round'] = {}

    def run_warmup(self):
        """
        Runs the warmup stage of training which is used to fill the agents memory.

        The agent uses it's rule-based policy to make actions. The agent's memory is filled as this runs.
        Loop terminates when the size of the memory is equal to WARMUP_MEM or when the memory buffer is full.
        """

        # TODO: change print with logger
        print('Warmup Started...')
        total_step = 0
        while total_step != self.warmup_mem and not self.dialogue_system.dqn_agent.is_memory_full():
            # Reset episode
            self.dialogue_system.reset()
            done = False

            # Get initial state from state tracker
            state = self.dialogue_system.state_tracker.get_state()
            while not done:
                next_state, _, done, _ = self.dialogue_system.run_round(use_rule=True)
                total_step += 1
                state = next_state

        print('...Warmup Ended')

    def run_train(self):
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
            self.dialogue_system.reset()
            episode += 1
            done = False
            rounds = 0

            # use sigma for partial switch to agent
            use_rule = False
            a = -float(self.sigma_init - self.sigma_stop) / self.sigma_decay
            b = float(self.sigma_init)
            sigma = max(self.sigma_stop, a * float(episode) + b)
            if sigma > random.random():
                use_rule = True

            while not done:
                _, reward, done, success = self.dialogue_system.run_round(use_rule=use_rule)
                period_reward_total += reward
                rounds += 1

            period_success_total += success
            period_round_total += rounds

            # Train
            if episode % self.train_freq == 0:
                # evaluate metrics
                self.performance_metrics['train']['success_rate'][episode] = period_success_total / self.train_freq
                self.performance_metrics['train']['avg_reward'][episode] = period_reward_total / self.train_freq
                self.performance_metrics['train']['avg_round'][episode] = period_round_total / self.train_freq

                # Check success rate
                success_rate = period_success_total / self.train_freq
                avg_reward = period_reward_total / self.train_freq

                # Flush
                if success_rate >= success_rate_best and success_rate >= self.success_rate_threshold:
                    self.dialogue_system.dqn_agent.empty_memory()

                # Update current best success rate
                if success_rate > success_rate_best:
                    print(f'Episode: {episode} NEW BEST SUCCESS RATE: {success_rate} Avg Reward: {avg_reward}')
                    success_rate_best = success_rate
                    self.dialogue_system.dqn_agent.save_weights(episode)
                period_success_total = 0
                period_reward_total = 0
                period_round_total = 0

                # Copy
                self.dialogue_system.dqn_agent.copy()

                # Train
                self.dialogue_system.dqn_agent.train()

                # Test on the actual weights
                # self.test(episode)

        print('...Training Ended')
        self.save_performance_records()

    def train(self):
        self.run_warmup()
        self.run_train()

    def save_performance_records(self):
        """Save performance numbers."""
        try:
            json.dump(self.performance_metrics, open(self.path, "w"), indent=2)
            print(f'saved model in {self.path}')
        except Exception as e:
            print(f'Error: Writing model fails: {self.path}')
            print(e)


# # Load params json into dict
# ROOT = 'params/'
# CHECKPOINTS = 'checkpoints/'
# PARAMS_FILE_PATH = []

# for file in os.listdir(ROOT):
#     if file == 'params_agt_boltz_4.json' or file == 'params_const.json':
#         if not os.path.exists(os.path.join(CHECKPOINTS, file.replace('params', 'performance'))):
#             PARAMS_FILE_PATH.append(os.path.join(ROOT, file))

# print(PARAMS_FILE_PATH)
# for run in range(10):
#     for path in PARAMS_FILE_PATH:
#         print(path)
#         params = {}
#         with open(path) as f:
#             params = json.load(f)

#         if not os.path.exists(params['agent']['save_weights_file_path'].replace('.h5', f'/{run}')):
#             os.makedirs(params['agent']['save_weights_file_path'].replace('.h5', f'/{run}'))
#         params['agent']['performance_path'] = params['agent']['performance_path'].replace('.json', f'_run{run}.json')
#         params['agent']['save_weights_file_path'] = params['agent']['save_weights_file_path'].replace('.h5', f'/{run}.h5')

#         if not os.path.exists(params['agent']['performance_path']):
#             dialogue_system = DialogueSystem(params)

#             dialogue_system.train()
