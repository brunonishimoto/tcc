import argparse
import json
from runners import Trainer
import os
import multiprocessing as mp

# Load config json into dict
ROOT = 'config/'
CHECKPOINTS = 'checkpoints/'
CONFIG_FILE_PATH = []

for file in os.listdir(ROOT):
    if file != 'config_eps.json' and file != 'config_softmax_slack.json' and file != 'test.json':
        if file == 'config_softmax_belief_drqn.json' or file == "config_softmax_belief_drqn1.json":
            CONFIG_FILE_PATH.append(os.path.join(ROOT, file))

print(CONFIG_FILE_PATH)
def train_model(path):
    for run in range(3):
        # for path in CONFIG_FILE_PATH:
        print(path)
        config = {}
        with open(path) as f:
            config = json.load(f)

        config['agent']['save_weights_file_path'] = config['agent']['save_weights_file_path'].replace('.h5', f'{run}.h5')
        if not os.path.exists(os.path.split(config['agent']['save_weights_file_path'])[0]):
            os.makedirs(os.path.split(config['agent']['save_weights_file_path'])[0])
        config['run']['performance_path'] = config['run']['performance_path'].replace('.json', f'_{run}.json')

        trainer = Trainer(config)

        trainer.run()

pool = mp.Pool(mp.cpu_count())
pool.map_async(train_model, CONFIG_FILE_PATH)
pool.close()
pool.join()
