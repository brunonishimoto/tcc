from .nlu_baseline import NLUBaseline
from .nlu_slot_gated import NLUSlotGated

def load(config):
    cls_name = config["nlu"]["name"]
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception(f"No such model: {cls_name}")
