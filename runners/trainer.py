import argparse
import json
import os
import random
import collections
from dialogue_system import DialogueSystem
from recorder import Recorder


class Trainer:

    def __init__(self, config):

        # Load run config
        run_dict = config['run']

        self.warmup_mem = run_dict['warmup_mem']
        self.num_ep_run = run_dict['num_ep_run']
        self.train_freq = run_dict['train_freq']

        self.max_round_num = run_dict['max_round_num']
        self.success_rate_threshold = run_dict['success_rate_threshold']

        # sigma parameter for using designer's knowledge
        self.sigma_init = run_dict['sigma_init']
        self.sigma_stop = run_dict['sigma_stop']
        self.sigma_decay = run_dict['sigma_decay']

        self.dialogue_system = DialogueSystem(config)
        self.recorder = Recorder(config)

        self.performance_metrics = collections.defaultdict(dict)
        self.performance_metrics['train']['success_rate'] = {}
        self.performance_metrics['train']['avg_reward'] = {}
        self.performance_metrics['train']['avg_round'] = {}

    def __run_warmup(self):
        """
        Runs the warmup stage of training which is used to fill the agents memory.

        The agent uses it's rule-based policy to make actions. The agent's memory is filled as this runs.
        Loop terminates when the size of the memory is equal to WARMUP_MEM or when the memory buffer is full.
        """

        # TODO: change print with logger
        print('Warmup Started...')
        total_step = 0
        episode = 0
        while total_step != self.warmup_mem and not self.dialogue_system.agent.is_memory_full():
            # Reset episode
            self.dialogue_system.reset(episode)
            done = False

            while not done:
                next_state, _, done, _ = self.dialogue_system.run_round(use_rule=True)
                total_step += 1
                state = next_state

            episode += 1

        print('...Warmup Ended')

    def __run_train(self):
        """
        Runs the loop that trains the agent.

        Trains the agent on the goal-oriented chatbot task. Training of the agent's neural network occurs
        every episode that TRAIN_FREQ is a multiple of. Terminates when the episode reaches NUM_EP_TRAIN.
        """

        print('Training Started...')
        episode = 0
        period_metrics = {'reward': 0, 'success': 0, 'round': 0}
        best_success_rate = 0.0

        while episode < self.num_ep_run:
            self.dialogue_system.reset(episode)
            done = False
            rounds = 0

            while not done:
                # Update sigma for a soft transition
                sigma = self.__update_sigma(episode)
                use_rule = True if random.random() <= sigma else False

                _, reward, done, success = self.dialogue_system.run_round(episode=episode, use_rule=use_rule)
                period_metrics['reward'] += reward
                rounds += 1

            period_metrics['success'] += success
            period_metrics['round'] += rounds


            # Train
            if episode % self.train_freq == 0:

                # evaluate metrics
                self.performance_metrics['train']['success_rate'][episode] = period_metrics['success'] / self.train_freq
                self.performance_metrics['train']['avg_reward'][episode] = period_metrics['reward'] / self.train_freq
                self.performance_metrics['train']['avg_round'][episode] = period_metrics['round'] / self.train_freq

                # Check success rate
                success_rate = period_metrics['success'] / self.train_freq
                avg_reward = period_metrics['reward'] / self.train_freq


                # Flush
                if success_rate >= best_success_rate and success_rate >= self.success_rate_threshold:
                    self.dialogue_system.agent.empty_memory()

                # Update current best success rate
                if success_rate > best_success_rate:
                    print(f'Episode: {episode} NEW BEST SUCCESS RATE: {success_rate} Avg Reward: {avg_reward}')
                    best_success_rate = success_rate
                    self.dialogue_system.agent.save_weights()

                period_metrics['success'] = 0
                period_metrics['reward'] = 0
                period_metrics['round'] = 0

                # Copy
                self.dialogue_system.agent.copy()

                # Train
                self.dialogue_system.agent.train()


                # Test on the actual weights
                # self.test(episode)

            episode += 1

        print('...Training Ended')
        self.save_performance_records()

    def __update_sigma(self, episode):
        # use sigma for partial switch to agent
        a = -float(self.sigma_init - self.sigma_stop) / self.sigma_decay
        b = float(self.sigma_init)
        sigma = max(self.sigma_stop, a * float(episode) + b)
        return sigma

    def run(self):
        self.__run_warmup()
        self.__run_train()

    # TODO: remove this function, nad use the Recorder class or a util function
    def save_performance_records(self):
        """Save performance numbers."""
        self.path = 'teste.json'
        try:
            json.dump(self.performance_metrics, open(self.path, "w"), indent=2)
            print(f'saved model in {self.path}')
        except Exception as e:
            print(f'Error: Writing model fails: {self.path}')
            print(e)
