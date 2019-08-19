from .real_user import RealUser
from .rule_based_user_simulator import RuleBasedUserSimulator

def load(config):
    cls_name = config["user"]["name"]
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such model: {}".format(cls_name))
