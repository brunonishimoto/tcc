from .dqn_agent import DQNAgent
from .dqn_epsilon_decay import DQNEpsilonDecay
from .dqn_softmax import DQNSoftmax

def load(config):
    cls_name = config["agent"]["name"]
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception(f"No such agent: {cls_name}")
