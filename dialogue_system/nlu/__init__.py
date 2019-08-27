from .nlu import NLU
from .bi_lstm import biLSTM
from .lstm import lstm

def load(config):
    cls_name = config["nlu"]["name"]
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception(f"No such model: {cls_name}")
