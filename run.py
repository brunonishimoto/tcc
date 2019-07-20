import argparse
import json
from trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_path', dest='params_path', type=str, default='')
    parser.add_argument('--train', dest='train', type=int, default=0, help='Either to train (1) or to test (0)')
    args = parser.parse_args()
    args = vars(args)

    # Load params json into dict
    PARAMS_FILE_PATH = 'params/params_const.json'
    if len(args['params_path']) > 0:
        params_file = args['params_path']
    else:
        params_file = PARAMS_FILE_PATH

    with open(params_file) as f:
        params = json.load(f)

    trainer = Trainer(params)
    trainer.train()
    # if (args['train'] == 1):
    #     dialogue_system.train()
    # elif (args['train'] == 0):
    #     dialogue_system.test()
