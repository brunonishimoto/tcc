import argparse
import json
from dialogue_system import DialogueSystem

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_path', dest='params_path', type=str, default='')
    parser.add_argument('--train', dest='train', type=int, default=0, help='Either to train (1) or to test (0)')
    args = parser.parse_args()
    args = vars(args)

    # Load params json into dict
    PARAMS_FILE_PATH = 'dialogue_system/params/params.json'
    if len(args['params_path']) > 0:
        params_file = args['params_path']
    else:
        params_file = PARAMS_FILE_PATH

    with open(params_file) as f:
        params = json.load(f)

    dialogue_system = DialogueSystem(params)

    if (args['train'] == 1):
        dialogue_system.train()
    elif (args['train'] == 0):
        dialogue_system.test()
