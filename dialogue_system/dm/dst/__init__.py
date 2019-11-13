from .state_tracker import StateTracker
from .belief_state_tracker import BeliefStateTracker
from .belief_state_tracker1 import BeliefStateTracker1
from .belief_state_tracker_new import BeliefStateTrackerNew
from .belief_state_tracker_probs import BeliefStateTrackerProbs

def load(config):
    cls_name = config["dst"]["name"]
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception(f"No such agent: {cls_name}")
