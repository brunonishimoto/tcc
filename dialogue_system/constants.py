"""

@author: Bruno Eidi Nishimoto <bruno_nishimoto@hotmail.com>
"""


########################################################################################################################
# Slots-related constants                                                                                              #
########################################################################################################################
# key for specifying the inform slots
INFORM_SLOTS = "inform_slots"
# key for specifying the request slots
REQUEST_SLOTS = "request_slots"
# key for specifying the history of all inform slots
HISTORY_SLOTS = "history_slots"
# key for specifying the history of all slots
REST_SLOTS = "rest_slots"
# key for specifying the task complete slot
TASK_COMPLETE_SLOT = "taskcomplete"
# key for specifying a proposed slot
PROPOSED_SLOT = "proposed_slots"

########################################################################################################################
# Agent training related constants                                                                                     #
########################################################################################################################

# key for specifying the simulation mode
SIMULATION_MODE = "simulation_mode"
# value for the semantic frame simulation mode
SEMANTIC_FRAME_SIMULATION_MODE = "semantic_frame_simulation_mode"
# flag indicating the mode of the dialogue system
IS_TRAINING = "is_training"
# key for specifying the maximal number of dialogue turns
MAX_NUM_TURNS = "max_num_turns"

########################################################################################################################
# User and Agent action related constants                                                                              #
########################################################################################################################
# key for specifying the intent (act) of the dialogue turn
INTENT = "intent"
# value for the unknown slots
UNKNOWN = "UNK"
# value for the placeholder slots
PLACEHOLDER = "PLACEHOLDER"
# key for specifying the speaker type (user or agent)
SPEAKER_TYPE = "speaker"
# value for the speaker key, when the user is the speaker
USR_SPEAKER_VAL = "User"
# value for the speaker key, when the agent is the speaker
AGT_SPEAKER_VAL = "Agent"
# key for specifying the turn number
ROUND = "round"

########################################################################################################################
# Knowledge Base related constants                                                                                     #
########################################################################################################################
# key for specifying a kb querying result where all of the constraints were matched
KB_MATCHING_ALL_CONSTRAINTS = "matching_all_constraints"

########################################################################################################################
# Dialog status related constants                                                                                      #
########################################################################################################################
# dialogue status
FAILED_DIALOG = -1
SUCCESS_DIALOG = 1
NO_OUTCOME_YET = 0

########################################################################################################################
# all dialogue acts                                                                                                    #
########################################################################################################################
# key for specifying request dialogue act
REQUEST = "request"
# key for specifying inform dialogue act
INFORM = "inform"
# key for specifying confirm question dialogue act
CONFIRM_QUESTION = "confirm_question"
# key for specifying confirm answer dialogue act
CONFIRM_ANSWER = "confirm_answer"
# key for specifying greeting dialogue act
GREETING = "greeting"
# key for specifying closing dialogue act
CLOSING = "closing"
# key for specifying multiple choice dialogue act
MULTIPLE_CHOICE = "multiple_choice"
# key for specifying thanks dialogue act
THANKS = "thanks"
# key for specifying welcome dialogue act
WELCOME = "welcome"
# key for specifying reject dialogue act
REJECT = "reject"
# key for specifying deny dialogue act
DENY = "deny"
# key for specifying not sure dialogue act
NOT_SURE = "not_sure"
# key for specifying done dialogue act
DONE = "done"
# key for specifying match found dialogue act
MATCH_FOUND = "match_found"

########################################################################################################################
#  Constraint Check                                                                                                    #
########################################################################################################################
CONSTRAINT_CHECK_FAILURE = 0
CONSTRAINT_CHECK_SUCCESS = 1

########################################################################################################################
#  Special Slot Values                                                                                                 #
########################################################################################################################
I_DO_NOT_CARE = "I do not care"
NO_MATCH = "no match available"
TICKET_AVAILABLE = "Ticket Available"
ANYTHING = "anything"
NOT_DONE = 'not done'
