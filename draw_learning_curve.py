'''
Created on Nov 3, 2016

draw a learning curve

@author: xiul
'''

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np


def read_performance_records(path):
    """ load the performance score (.json) file """

    data = json.load(open(path, 'rb'))
    for key in data['success_rate'].keys():
        if int(key) > -1:
            print("%s\t%s\t%s\t%s" % (key, data['success_rate'][key], data['avg_round'][key], data['avg_reward'][key]))


def load_performance_file(path):
    """ load the performance score (.json) file """

    data = json.load(open(path, 'rb'))
    numbers = {'x': [], 'success_rate': [], 'avg_round': [], 'avg_reward': []}
    keylist = [int(key) for key in data['success_rate'].keys()]
    keylist.sort()

    for key in keylist:
        if int(key) > -1:
            numbers['x'].append(int(key) / 100)
            numbers['success_rate'].append(data['success_rate'][str(key)])
            numbers['avg_round'].append(data['avg_round'][str(key)])
            numbers['avg_reward'].append(data['avg_reward'][str(key)])
    return numbers


def max_performance(path):
    """ return the max performance during training. """

    numbers = load_performance_file(path)
    max_acc = np.argmax(numbers['success_rate'])
    return {
        'success_rate': numbers['success_rate'][max_acc],
        'avg_round': numbers['avg_round'][max_acc],
        'avg_reward': numbers['avg_reward'][max_acc]
    }


def draw_learning_curve(numbers, metric):
    """ draw the learning curve """

    plt.xlabel('Simulation Epoch')
    plt.ylabel(metric)
    plt.title('Learning Curve')
    plt.grid(True)

    plt.plot(numbers['x'], numbers[metric], 'r', lw=1)
    plt.show()


def main(params):
    cmd = params['cmd']

    if cmd == 0:
        numbers = load_performance_file(params['result_file'])
        for metric in params['metrics']:
            draw_learning_curve(numbers, metric)
    elif cmd == 1:
        read_performance_records(params['result_file'])
    elif cmd == 2:
        print(max_performance(params['result_file']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cmd', dest='cmd', type=int, default=1, help='cmd')
    parser.add_argument('-m', '--metrics', dest='metrics', type=list, default=['success_rate', 'avg_round', 'avg_reward'],
                        help='the metrics to plot')

    parser.add_argument('--result_file', dest='result_file', type=str, default='checkpoints/performance.json',
                        help='path to the result file')

    args = parser.parse_args()
    params = vars(args)
    print(json.dumps(params, indent=2))

    main(params)
