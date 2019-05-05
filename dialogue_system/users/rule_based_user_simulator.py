import dialogue_system.dialogue_config as config
import dialogue_system.constants as const
import random
import copy


class RuleBasedUserSimulator:
    """Simulates a real user, to train the agent with reinforcement learning."""

    def __init__(self, goal_list, params, database):
        """
        The constructor for UserSimulator. Sets dialogue config variables.

        Parameters:
            goal_list (list): User goals loaded from file
            params (dict): Dict of params loaded from file
            database (dict): The database in the format dict(long: dict)
        """

        self.goal_list = goal_list
        self.max_round = params['run']['max_round_num']
        self.default_key = config.usersim_default_key
        # A list of REQUIRED to be in the first action inform keys
        self.init_informs = config.usersim_required_init_inform_keys
        self.no_query = config.no_query_keys

        # TEMP ----
        self.database = database
        # ---------

    def reset(self):
        """
        Resets the user sim. by emptying the state and returning the initial action.

        Returns:
            dict: The initial action of an episode
        """

        # Sample a random goal
        self.goal = self._sample_goal()
        # Add default slot to requests of goal
        self.goal[const.REQUEST_SLOTS][self.default_key] = const.UNKNOWN

        # Reset the user state
        self.state = {}
        # Add all inform slots informed by agent or user sim to this dict
        self.state[const.HISTORY_SLOTS] = {}
        # Any inform slots for the current user sim action, empty at start of turn
        self.state[const.INFORM_SLOTS] = {}
        # Current request slots the user sim wants to request
        self.state[const.REQUEST_SLOTS] = {}
        # Init. all informs and requests in user goal, remove slots as informs made by user or agent
        self.state[const.REST_SLOTS] = {}
        self.state[const.REST_SLOTS].update(self.goal[const.INFORM_SLOTS])
        self.state[const.REST_SLOTS].update(self.goal[const.REQUEST_SLOTS])
        self.state[const.INTENT] = ''

        # False for failure, true for success, init. to failure
        self.constraint_check = const.FAILED_DIALOG
        self.episode_over = False
        self.dialogue_status = const.NO_OUTCOME_YET

        return self._return_init_action()

    def _sample_goal(self):
        sample_goal = random.choice(self.goal_list)
        # print(f'-------------------------------------------------------\nUser Goal: {sample_goal}\n')
        return sample_goal

    def _return_init_action(self):
        """
        Returns the initial action of the episode.

        The initial action has an intent of request, required init. inform slots and a single request slot.

        Returns:
            dict: Initial user response
        """

        # The first dialogue intent (action) is always a request
        self.state[const.INTENT] = const.REQUEST

        if self.goal[const.INFORM_SLOTS]:
            # Pick all the required init. informs, and add if they exist in goal inform slots
            for inform_key in self.init_informs:
                if inform_key in self.goal[const.INFORM_SLOTS]:
                    self.state[const.INFORM_SLOTS][inform_key] = self.goal[const.INFORM_SLOTS][inform_key]
                    self.state[const.REST_SLOTS].pop(inform_key)
                    self.state[const.HISTORY_SLOTS][inform_key] = self.goal[const.INFORM_SLOTS][inform_key]
            # If nothing was added then pick a random one to add
            if not self.state[const.INFORM_SLOTS]:
                key, value = random.choice(list(self.goal[const.INFORM_SLOTS].items()))
                self.state[const.INFORM_SLOTS][key] = value
                self.state[const.REST_SLOTS].pop(key)
                self.state[const.HISTORY_SLOTS][key] = value

        # Now add a request, do a random one if something other than def. available
        self.goal[const.REQUEST_SLOTS].pop(self.default_key)
        if self.goal[const.REQUEST_SLOTS]:
            req_key = random.choice(list(self.goal[const.REQUEST_SLOTS].keys()))
        else:
            req_key = self.default_key
        self.goal[const.REQUEST_SLOTS][self.default_key] = const.UNKNOWN
        self.state[const.REQUEST_SLOTS][req_key] = const.UNKNOWN

        user_response = {}
        user_response[const.INTENT] = self.state[const.INTENT]
        user_response[const.REQUEST_SLOTS] = copy.deepcopy(self.state[const.REQUEST_SLOTS])
        user_response[const.INFORM_SLOTS] = copy.deepcopy(self.state[const.INFORM_SLOTS])

        return user_response

    def step(self, agent_action):
        """
        Return the response of the user sim. to the agent by using rules that simulate a user.

        Given the agent action craft a response by using deterministic rules that simulate (to some extent) a user.
        Some parts of the rules are stochastic. Check if the agent has succeeded or lost or still going.

        Parameters:
            agent_action (dict): The agent action that the user sim. responds to

        Returns:
            dict: User sim. response
            int: Reward
            bool: Done flag
            int: Success: -1, 0 or 1 for loss, neither win nor loss, win
        """

        # Assertions -----
        # No UNK in agent action informs
        for value in agent_action[const.INFORM_SLOTS].values():
            assert value != const.UNKNOWN
            assert value != const.PLACEHOLDER
        # No PLACEHOLDER in agent at all
        for value in agent_action[const.REQUEST_SLOTS].values():
            assert value != const.PLACEHOLDER
        # ----------------

        self.state[const.INFORM_SLOTS].clear()
        self.state[const.INTENT] = ''

        done = False
        success = const.NO_OUTCOME_YET
        # First check round num, if equal to max then fail
        if self.max_round > 0 and agent_action[const.ROUND] == self.max_round:
            done = True
            success = const.FAILED_DIALOG
            self.state[const.INTENT] = const.DONE
            self.state[const.REQUEST_SLOTS].clear()
        else:
            agent_intent = agent_action[const.INTENT]
            if agent_intent == const.REQUEST:
                self._response_to_request(agent_action)
            elif agent_intent == const.INFORM:
                self._response_to_inform(agent_action)
            elif agent_intent == const.MATCH_FOUND:
                self._response_to_match_found(agent_action)
            elif agent_intent == const.DONE:
                success = self._response_to_done()
                self.state[const.INTENT] = const.DONE
                self.state[const.REQUEST_SLOTS].clear()
                done = True

        # Assumptions -------
        # If request intent, then make sure request slots
        if self.state[const.INTENT] == const.REQUEST:
            assert self.state[const.REQUEST_SLOTS]
        # If inform intent, then make sure inform slots and NO request slots
        if self.state[const.INTENT] == const.INFORM:
            assert self.state[const.INFORM_SLOTS]
            assert not self.state[const.REQUEST_SLOTS]
        assert const.UNKNOWN not in self.state[const.INFORM_SLOTS].values()
        assert const.PLACEHOLDER not in self.state[const.REQUEST_SLOTS].values()
        # No overlap between rest and hist
        for key in self.state[const.REST_SLOTS]:
            assert key not in self.state[const.HISTORY_SLOTS]
        for key in self.state[const.HISTORY_SLOTS]:
            assert key not in self.state[const.REST_SLOTS]
        # All slots in both rest and hist should contain the slots for goal
        for inf_key in self.goal[const.INFORM_SLOTS]:
            assert self.state[const.HISTORY_SLOTS].get(inf_key, False) \
                or self.state[const.REST_SLOTS].get(inf_key, False)
        for req_key in self.goal[const.REQUEST_SLOTS]:
            assert self.state[const.HISTORY_SLOTS].get(req_key, False) \
                or self.state[const.REST_SLOTS].get(req_key, False), req_key
        # Anything in the rest should be in the goal
        for key in self.state[const.REST_SLOTS]:
            assert self.goal[const.INFORM_SLOTS].get(key, False) or self.goal[const.REQUEST_SLOTS].get(key, False)
        assert self.state[const.INTENT] != ''
        # -----------------------

        user_response = {}
        user_response[const.INTENT] = self.state[const.INTENT]
        user_response[const.REQUEST_SLOTS] = copy.deepcopy(self.state[const.REQUEST_SLOTS])
        user_response[const.INFORM_SLOTS] = copy.deepcopy(self.state[const.INFORM_SLOTS])

        reward = self._reward_function(success)

        return user_response, reward, done, True if success is 1 else False

    def _response_to_request(self, agent_action):
        """
        Augments the state in response to the agent action having an intent of request.

        There are 4 main cases for responding.

        Parameters:
            agent_action (dict): Intent of request with standard action format (including 'speaker': 'Agent' and
                                 'round_num': int)
        """

        # if the agent action cointains request slots
        if len(agent_action[const.REQUEST_SLOTS].keys()) > 0:

            # take the first request slot
            agent_request_key = list(agent_action[const.REQUEST_SLOTS].keys())[0]

            # First Case: if agent requests for something that is in the user sims goal inform slots, then inform it
            if agent_request_key in self.goal[const.INFORM_SLOTS]:
                self.state[const.INTENT] = const.INFORM
                self.state[const.INFORM_SLOTS][agent_request_key] = self.goal[const.INFORM_SLOTS][agent_request_key]

                # remove the inform slot from the rest slots
                self.state[const.REST_SLOTS].pop(agent_request_key, None)

                # add the inform slot to the history slots
                self.state[const.HISTORY_SLOTS][agent_request_key] = self.goal[const.INFORM_SLOTS][agent_request_key]

                self.state[const.REQUEST_SLOTS].clear()
            # Second Case: if the agent requests for something in user sims goal request slots and it has already been
            # informed, then inform it
            elif agent_request_key in self.goal[const.REQUEST_SLOTS] \
                    and agent_request_key in self.state[const.HISTORY_SLOTS] \
                    and agent_request_key not in self.state[const.REST_SLOTS]:

                self.state[const.INTENT] = const.INFORM
                self.state[const.INFORM_SLOTS][agent_request_key] = self.state[const.HISTORY_SLOTS][agent_request_key]
                self.state[const.REQUEST_SLOTS].clear()
            # Third Case: if the agent requests for something in the user sims goal request slots and it HASN'T been
            # informed, then request it with a random inform
            elif agent_request_key in self.goal[const.REQUEST_SLOTS] \
                    and agent_request_key in self.state[const.REST_SLOTS]:
                self.state[const.REQUEST_SLOTS].clear()
                self.state[const.INTENT] = const.REQUEST
                self.state[const.REQUEST_SLOTS][agent_request_key] = const.UNKNOWN

                rest_informs = {}
                for key, value in list(self.state[const.REST_SLOTS].items()):
                    if key in self.goal[const.INFORM_SLOTS]:
                        rest_informs[key] = value

                if rest_informs:
                    key_choice, value_choice = random.choice(list(rest_informs.items()))
                    self.state[const.INFORM_SLOTS][key_choice] = value_choice
                    self.state[const.REST_SLOTS].pop(key_choice)
                    self.state[const.HISTORY_SLOTS][key_choice] = value_choice
            # Fourth and Final Case: otherwise the user sim does not care about the slot being requested, then inform
            # 'anything' as the value of the requested slot
            else:
                assert agent_request_key not in self.state[const.REST_SLOTS]
                self.state[const.INTENT] = const.INFORM
                self.state[const.INFORM_SLOTS][agent_request_key] = const.ANYTHING
                self.state[const.REQUEST_SLOTS].clear()
                self.state[const.HISTORY_SLOTS][agent_request_key] = const.ANYTHING
        # if there are no request slot in the agent action (problably we should not be here),
        # take a random slot from rest
        else:
            if len(self.state[const.REST_SLOTS]) > 0:
                random_slot = random.choice(self.state[const.REST_SLOTS])

                # if the random slot is in inform slots
                if random_slot in self.goal[const.INFORM_SLOTS]:
                    self.state[const.INFORM_SLOTS][random_slot] = self.goal[const.INFORM_SLOTS][random_slot]

                    self.state[const.REST_SLOTS].pop(random_slot)
                    self.state[const.INTENT] = const.INFORM
                # if the random slot is in the request slots
                elif random_slot in self.goal[const.REQUEST_SLOTS]:
                    self.state[const.REQUEST_SLOTS][random_slot] = self.goal[const.REQUEST_SLOTS][random_slot]

                    self.state[const.INTENT] = const.REQUEST

    def _response_to_inform(self, agent_action):
        """
        Augments the state in response to the agent action having an intent of inform.

        There are 2 main cases for responding. Add the agent inform slots to history slots,
        and remove the agent inform slots from the rest and request slots.

        Parameters:
            agent_action (dict): Intent of inform with standard action format (including 'speaker': 'Agent' and
                                 'round_num': int)
        """

        agent_inform_key = list(agent_action[const.INFORM_SLOTS].keys())[0]
        agent_inform_value = agent_action[const.INFORM_SLOTS][agent_inform_key]

        assert agent_inform_key != self.default_key

        # Add all informs (by agent too) to hist slots
        self.state[const.HISTORY_SLOTS][agent_inform_key] = agent_inform_value
        # Remove from rest slots if in it
        self.state[const.REST_SLOTS].pop(agent_inform_key, None)
        # Remove from request slots if in it
        self.state[const.REQUEST_SLOTS].pop(agent_inform_key, None)

        # If agent informs something that is in user goal informs
        if agent_inform_key in self.goal[const.INFORM_SLOTS]:
            # if the value doesn't match the user goal inform value, then inform the correct value
            if agent_inform_value != self.goal[const.INFORM_SLOTS].get(agent_inform_key, agent_inform_value):
                self.state[const.INTENT] = const.INFORM
                self.state[const.INFORM_SLOTS][agent_inform_key] = self.goal[const.INFORM_SLOTS][agent_inform_key]
                self.state[const.REQUEST_SLOTS].clear()
                self.state[const.HISTORY_SLOTS][agent_inform_key] = self.goal[const.INFORM_SLOTS][agent_inform_key]
            # if it match the value take a random action
            else:
                # - If anything in state requests then request it
                if self.state[const.REQUEST_SLOTS]:
                    self.state[const.INTENT] = const.REQUEST
                # - Else if something to say in rest slots, pick something
                elif self.state[const.REST_SLOTS]:
                    def_in = self.state[const.REST_SLOTS].pop(self.default_key, False)
                    if self.state[const.REST_SLOTS]:
                        key, value = random.choice(list(self.state[const.REST_SLOTS].items()))
                        if value != const.UNKNOWN:
                            self.state[const.INTENT] = const.INFORM
                            self.state[const.INFORM_SLOTS][key] = value
                            self.state[const.REST_SLOTS].pop(key)
                            self.state[const.HISTORY_SLOTS][key] = value
                        else:
                            self.state[const.INTENT] = const.REQUEST
                            self.state[const.REQUEST_SLOTS][key] = const.UNKNOWN
                    else:
                        self.state[const.INTENT] = const.REQUEST
                        self.state[const.REQUEST_SLOTS][self.default_key] = const.UNKNOWN
                    if def_in == const.UNKNOWN:
                        self.state[const.REST_SLOTS][self.default_key] = const.UNKNOWN
                else:
                    self.state[const.INTENT] = const.THANKS
        # if the agent informs something that is not in the user goal inform slots
        else:
            # chose from the request slots
            if self.state[const.REQUEST_SLOTS]:
                def_in = self.state[const.REST_SLOTS].pop(self.default_key, False)
                if self.state[const.REQUEST_SLOTS]:
                    key, value = random.choice(list(self.state[const.REQUEST_SLOTS].items()))
                    self.state[const.INTENT] = const.REQUEST
                    self.state[const.REQUEST_SLOTS][key] = const.UNKNOWN
                else:
                    self.state[const.INTENT] = const.REQUEST
                    self.state[const.REQUEST_SLOTS][self.default_key] = const.UNKNOWN
                if def_in == const.UNKNOWN:
                    self.state[const.REST_SLOTS][self.default_key] = const.UNKNOWN
            elif self.state[const.REST_SLOTS]:
                def_in = self.state[const.REST_SLOTS].pop(self.default_key, False)
                if self.state[const.REST_SLOTS]:
                    key, value = random.choice(list(self.state[const.REST_SLOTS].items()))
                    if value != const.UNKNOWN:
                        self.state[const.INTENT] = const.INFORM
                        self.state[const.INFORM_SLOTS][key] = value
                        self.state[const.REST_SLOTS].pop(key)
                        self.state[const.HISTORY_SLOTS][key] = value
                    else:
                        self.state[const.INTENT] = const.REQUEST
                        self.state[const.REQUEST_SLOTS][key] = const.UNKNOWN
                else:
                    self.state[const.INTENT] = const.REQUEST
                    self.state[const.REQUEST_SLOTS][self.default_key] = const.UNKNOWN
                if def_in == const.UNKNOWN:
                    self.state[const.REST_SLOTS][self.default_key] = const.UNKNOWN
            else:
                self.state[const.INTENT] = const.THANKS

    def _response_to_match_found(self, agent_action):
        """
        Augments the state in response to the agent action having an intent of match_found.

        Check if there is a match in the agent action that works with the current goal.

        Parameters:
            agent_action (dict): Intent of match_found with standard action format (including 'speaker': 'Agent' and
                                 'round_num': int)
        """

        agent_informs = agent_action[const.INFORM_SLOTS]

        self.state[const.INTENT] = const.THANKS
        self.constraint_check = const.SUCCESS_DIALOG

        assert self.default_key in agent_informs
        self.state[const.REST_SLOTS].pop(self.default_key, None)
        self.state[const.HISTORY_SLOTS][self.default_key] = str(agent_informs[self.default_key])
        self.state[const.REQUEST_SLOTS].pop(self.default_key, None)

        if agent_informs[self.default_key] == const.NO_MATCH:
            self.constraint_check = const.FAILED_DIALOG

        # Check to see if all goal informs are in the agent informs, and that the values match
        for key, value in self.goal[const.INFORM_SLOTS].items():
            assert value is not None
            # For items that cannot be in the queries don't check to see if they are in the agent informs here
            if key in self.no_query:
                continue
            # Will return true if key not in agent informs OR if value does not match value of agent informs[key]
            if value != agent_informs.get(key, None):
                self.constraint_check = const.FAILED_DIALOG
                break

        if self.constraint_check == const.FAILED_DIALOG:
            self.state[const.INTENT] = const.REJECT
            self.state[const.REQUEST_SLOTS].clear()
            self.state[const.INFORM_SLOTS].clear()

    def _response_to_done(self):
        """
        Augments the state in response to the agent action having an intent of done.

        If the constraint_check is SUCCESS and both the rest and request slots of the state are empty for the agent
        to succeed in this episode/conversation.

        Returns:
            int: Success: -1, 0 or 1 for loss, neither win nor loss, win
        """

        if self.constraint_check == const.FAILED_DIALOG:
            return const.FAILED_DIALOG

        if not self.state[const.REST_SLOTS]:
            assert not self.state[const.REQUEST_SLOTS]
        if self.state[const.REST_SLOTS]:
            return const.FAILED_DIALOG

        # TEMP: ----
        assert self.state[const.HISTORY_SLOTS][self.default_key] != const.NO_MATCH

        match = copy.deepcopy(self.database[int(self.state[const.HISTORY_SLOTS][self.default_key])])

        for key, value in self.goal[const.INFORM_SLOTS].items():
            assert value is not None
            if key in self.no_query:
                continue
            if value != match.get(key, None):
                assert True is False, 'match: {}\ngoal: {}'.format(match, self.goal)
                break
        # ----------

        return const.SUCCESS_DIALOG

    def _reward_function(self, success):
        """
        Return the reward given the success.

        Return -1 + -max_round if success is FAIL, -1 + 2 * max_round if success is SUCCESS and -1 otherwise.

        Parameters:
            success (int)

        Returns:
            int: Reward
        """

        reward = -1
        if success == const.FAILED_DIALOG:
            reward += -self.max_round
        elif success == const.SUCCESS_DIALOG:
            reward += 2 * self.max_round
        return reward
