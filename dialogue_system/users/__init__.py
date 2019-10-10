from .real_user_cmd import RealUserCMD
from .rule_based_user_simulator import RuleBasedUserSimulator
from .real_user_slack import RealUserSlack

def load(config):
    cls_name = config["user"]["name"]
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such model: {}".format(cls_name))
