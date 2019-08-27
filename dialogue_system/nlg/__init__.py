from .nlg import NLG
from .utils import *

def load(config):
    cls_name = config["nlg"]["name"]
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception(f"No such model: {cls_name}")
