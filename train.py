import argparse
import json
from dialogue_system import DialogueSystem
import os

# Load params json into dict
ROOT = 'params/'
CHECKPOINTS = 'checkpoints/'
PARAMS_FILE_PATH = []

for file in os.listdir(ROOT):
    if not os.path.exists(os.path.join(CHECKPOINTS, file.replace('params', 'performance'))):
        if (file == 'params100_1.json' or file == 'params100_5.json'):
            PARAMS_FILE_PATH.append(os.path.join(ROOT, file))


print(PARAMS_FILE_PATH)
for run in range(10):
    for path in PARAMS_FILE_PATH:
        print(path)
        params = {}
        with open(path) as f:
            params = json.load(f)

        if not os.path.exists(params['agent']['save_weights_file_path'].replace('.h5', f'/{run}')):
            os.makedirs(params['agent']['save_weights_file_path'].replace('.h5', f'/{run}'))
        params['agent']['performance_path'] = params['agent']['performance_path'].replace('.json', f'run{run}.json')
        params['agent']['save_weights_file_path'] = params['agent']['save_weights_file_path'].replace('.h5', f'/{run}.h5')
        dialogue_system = DialogueSystem(params)

        dialogue_system.train()
