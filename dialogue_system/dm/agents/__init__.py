from .dqn_agent import DQNAgent
from .drqn_agent import DRQNAgent
from .dqn_epsilon_decay import DQNEpsilonDecay
from .dqn_softmax import DQNSoftmax
from .drqn_softmax import DRQNSoftmax

def load(config):
    cls_name = config["agent"]["name"]
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception(f"No such agent: {cls_name}")
