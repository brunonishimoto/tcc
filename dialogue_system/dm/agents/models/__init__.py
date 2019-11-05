from .dqn_model import DQNModel
from .drqn_model import DRQNModel
from .drqn_model1 import DRQNModel1

def load(config):
    cls_name = config["model"]["name"]
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception(f"No such model: {cls_name}")
