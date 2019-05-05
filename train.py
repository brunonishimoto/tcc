import argparse
import json
from dialogue_system import DialogueSystem
import os

# Load params json into dict
ROOT = 'dialogue_system/params/'
CHECKPOINTS = 'checkpoints/'
PARAMS_FILE_PATH = []

for file in os.listdir(ROOT):
    if not os.path.exists(os.path.join(CHECKPOINTS, file.replace('params', 'performance'))):
        PARAMS_FILE_PATH.append(os.path.join(ROOT, file))


print(PARAMS_FILE_PATH)
for path in PARAMS_FILE_PATH:
    params = {}
    with open(path) as f:
        params = json.load(f)

    dialogue_system = DialogueSystem(params)

    dialogue_system.train()
