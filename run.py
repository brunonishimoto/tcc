import argparse
import json
import os
from trainer import Trainer
from tester import Tester

PARAMS_FILE_ROOT = 'params/'

def run(params_file='params.json', train=True):
    # Load params json into dict
    params_file_path = os.path.join(PARAMS_FILE_ROOT, params_file)

    params = None
    with open(params_file_path) as f:
        params = json.load(f)

    if train:
        trainer = Trainer(params)
        trainer.train()
    else:
        tester = Tester(params)
        tester.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_path', dest='params_path', type=str, default='params.json')
    parser.add_argument('--train', dest='train', type=int, default=0, help='Either to train (1) or to test (0)')
    args = parser.parse_args()
    args = vars(args)

    run(args['params_path'], args['train'])

