import argparse
import json
from dialogue_system import DialogueSystem
import os

# Load params json into dict
ROOT = 'params/'
CHECKPOINTS = 'checkpoints/'
PARAMS_FILE_PATH = []

for file in os.listdir(ROOT):
    if file == 'params_agt_boltz_4.json' or file == 'params_const.json':
        if not os.path.exists(os.path.join(CHECKPOINTS, file.replace('params', 'performance'))):
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
        params['agent']['performance_path'] = params['agent']['performance_path'].replace('.json', f'_run{run}.json')
        params['agent']['save_weights_file_path'] = params['agent']['save_weights_file_path'].replace('.h5', f'/{run}.h5')

        if not os.path.exists(params['agent']['performance_path']):
            dialogue_system = DialogueSystem(params)

            dialogue_system.train()
