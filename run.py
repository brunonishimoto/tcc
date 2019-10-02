import argparse
import json
import os
import runners

PARAMS_FILE_ROOT = 'config/'

def run(config_file='config.json'):
    # Load config json into dict
    config = None
    with open(os.path.join(PARAMS_FILE_ROOT, config_file)) as f:
        config = json.load(f)

    runner = runners.load(config)

    runner.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', dest='config_file', type=str, default='config_softmax.json')
    args = parser.parse_args()
    args = vars(args)

    run(args['config_file'])
