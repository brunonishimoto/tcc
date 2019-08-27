import collections
from dialogue_system import DialogueSystem
from utils.util import save_json_file


class Tester:

    def __init__(self, params):

        # Load run params
        run_dict = params['run']

        self.num_ep_test = run_dict['num_ep_test']

        self.performance_path = run_dict['performance_path']

        self.performance_metrics = collections.defaultdict(dict)

        self.dialogue_system = DialogueSystem(params)

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
        save_json_file(self.performance_path, self.performance_metrics)
