import argparse
import json
import collections
from dialogue_system import DialogueSystem
from recorder import Recorder


class Tester:

    def __init__(self, params):

        # Load run params
        run_dict = params['run']

        self.num_ep_test = run_dict['num_ep_test']

        self.performance_metrics = collections.defaultdict(dict)

        self.dialogue_system = DialogueSystem(params)
        self.recorder = Recorder(params)

    def run(self):
        """
        Runs the loop that tests the agent.

        Tests the agent on the goal-oriented chatbot task. Only for evaluating a trained agent.
        Terminates when the episode reaches NUM_EP_TEST.
        """

        print('Testing Started...')
        episode = 0
        period_metrics = {'reward': 0, 'success': 0, 'round': 0}
        period_metrics['reward'] = 0
        period_metrics['success'] = 0
        period_metrics['round'] = 0

        while episode < self.num_ep_test:
            self.dialogue_system.reset(episode, train=False)
            done = False

            success = False
            rounds = 0
            while not done:
                _, reward, done, success = self.dialogue_system.run_round(use_rule=False, train=False)
                period_metrics['reward'] += reward
                rounds += 1

            period_metrics['success'] += success
            period_metrics['round'] += rounds

            episode += 1

            # print(f'Episode: {episode} Success: {success} Reward: {ep_reward}')

        self.performance_metrics['test']['success_rate'] = period_metrics['success'] / self.num_ep_test
        self.performance_metrics['test']['avg_reward'] = period_metrics['reward'] / self.num_ep_test
        self.performance_metrics['test']['avg_round'] = period_metrics['round'] / self.num_ep_test

        print('...Testing Ended')
        self.save_performance_records()

    # TODO: remove this function, nad use the Recorder class or a util function
    def save_performance_records(self):
        """Save performance numbers."""
        try:
            json.dump(self.performance_metrics, open(self.path, "w"), indent=2)
            print(f'saved model in {self.path}')
        except Exception as e:
            print(f'Error: Writing model fails: {self.path}')
            print(e)

