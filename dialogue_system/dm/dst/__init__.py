from .state_tracker import StateTracker

def load(config):
    cls_name = config["dst"]["name"]
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception(f"No such agent: {cls_name}")
