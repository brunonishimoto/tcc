from users import RealUser, RuleBasedUserSimulator
from users.error_model_controller import ErrorModelController
from dm.agents import DQNAgent
from dm.dst import StateTracker
from utils.util import remove_empty_slots

import pickle
import argparse
import json
import math


if __name__ == "__main__":
    # Can provide params file path in args OR run it as is and change 'PARAMS_FILE_PATH' below
    # 1) In terminal: python test.py --params_path "params/params.json"
    # 2) Run this file as is
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_path', dest='params_path', type=str, default='')
    args = parser.parse_args()
    params = vars(args)

    # Load params json into dict
    PARAMS_FILE_PATH = 'params/params.json'
    if len(params['params_path']) > 0:
        params_file = params['params_path']
    else:
        params_file = PARAMS_FILE_PATH

    with open(params_file) as f:
        params = json.load(f)

    # Load file path params
    file_path_dict = params['db_file_paths']
    DATABASE_FILE_PATH = file_path_dict['database']
    DICT_FILE_PATH = file_path_dict['dict']
    USER_GOALS_FILE_PATH = file_path_dict['user_goals']

    # Load run params
    run_dict = params['run']
    USE_USERSIM = run_dict['usersim']
    NUM_EP_TEST = run_dict['num_ep_run']
    MAX_ROUND_NUM = run_dict['max_round_num']

    # Load movie DB
    # Note: If you get an unpickling error here then run 'pickle_converter.py' and it should fix it
    database = pickle.load(open(DATABASE_FILE_PATH, 'rb'), encoding='latin1')

    # Clean DB
    remove_empty_slots(database)

    # Load movie dict
    db_dict = pickle.load(open(DICT_FILE_PATH, 'rb'), encoding='latin1')

    # Load goal file
    user_goals = pickle.load(open(USER_GOALS_FILE_PATH, 'rb'), encoding='latin1')

    # Init. Objects
    if USE_USERSIM:
        user = RuleBasedUserSimulator(user_goals, params, database)
    else:
        user = RealUser(params)
    emc = ErrorModelController(db_dict, params)
    state_tracker = StateTracker(database, params)
    dqn_agent = DQNAgent(state_tracker.get_state_size(), params)


def test_run():
    """
    Runs the loop that tests the agent.

    Tests the agent on the goal-oriented chatbot task. Only for evaluating a trained agent. Terminates when the episode
    reaches NUM_EP_TEST.

    """

    print('Testing Started...')
    episode = 0
    while episode < NUM_EP_TEST:
        episode_reset()
        episode += 1
        ep_reward = 0
        done = False
        # Get initial state from state tracker
        state = state_tracker.get_state()
        while not done:
            # Agent takes action given state tracker's representation of dialogue
            agent_action_index, agent_action = dqn_agent.get_action(state)
            # Update state tracker with the agent's action
            state_tracker.update_state_agent(agent_action)
            # User takes action given agent action
            user_action, reward, done, success = user.step(agent_action)
            ep_reward += reward
            if not done:
                # Infuse error into semantic frame level of user action
                emc.infuse_error(user_action)
            # Update state tracker with user action
            state_tracker.update_state_user(user_action)
            # Grab "next state" as state
            state = state_tracker.get_state(done)
        print('Episode: {} Success: {} Reward: {}'.format(episode, success, ep_reward))
    print('...Testing Ended')


def episode_reset():
    """Resets the episode/conversation in the testing loop."""

    # First reset the state tracker
    state_tracker.reset()
    # Then pick an init user action
    user_action = user.reset()
    # Infuse with error
    emc.infuse_error(user_action)
    # And update state tracker
    state_tracker.update_state_user(user_action)
    # Finally, reset agent
    dqn_agent.reset()


test_run()
