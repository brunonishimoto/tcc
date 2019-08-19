from .tester import Tester
from .trainer import Trainer

def load(config):
    cls_name = config["runner"]["name"]
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such model: {}".format(cls_name))
