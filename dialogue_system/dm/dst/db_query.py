import dialogue_system.dialogue_config as cfg
import dialogue_system.constants as const
from collections import defaultdict
import copy


class DBQuery:
    """Queries the database for the state tracker."""

    def __init__(self, database):
        """
        The constructor for DBQuery.

        Parameters:
            database (dict): The database in the format dict(long: dict)
        """

        self.database = database
        self.no_query = cfg.no_query_keys
        self.match_key = cfg.usersim_default_key

        self.cached_db_slot = defaultdict(dict)
        self.cached_db = defaultdict(dict)

    def fill_inform_slot(self, inform_slots_to_be_filled, current_slots):
        """
        Given the current informs/constraints fill the informs that need to be filled with values from the database.

        Searches through the database to fill the inform slots with PLACEHOLDER with values that work given the current
        constraints of the current episode.

        Parameters:
            inform_slots_to_be_filled (dict): Inform slots to fill with values
            current_slots (dict): Current inform slots with values from the StateTracker

        Returns:
            dict: inform_slots_to_be_filled filled with values
        """

        # For this simple system only one inform slot should ever passed in
        # assert len(inform_slots_to_be_filled) == 1

        # key = list(inform_slots_to_be_filled.keys())[0]

        # # This removes the inform we want to fill from the current informs if it is present in the current informs
        # # so it can be re-queried
        # current_informs = copy.deepcopy(current_slots)
        # current_informs.pop(key, None)

        # # db_results is a dict of dict in the same exact format as the db, it is just a subset of the db
        # db_results = self.get_db_results(current_informs)

        # filled_inform = {}
        # values_dict = self._count_slot_values(key, db_results)
        # if values_dict:
        #     # Get key with max value (ie slot value with highest count of available results)
        #     filled_inform[key] = max(values_dict, key=values_dict.get)
        # else:
        #     filled_inform[key] = const.NO_MATCH

        # return filled_inform
        """ Takes unfilled inform slots and current_slots, returns dictionary of filled informed slots (with values)

        Arguments:
        inform_slots_to_be_filled   --  Something that looks like {starttime:None, theater:None} where starttime and theater are slots that the agent needs filled
        current_slots               --  Contains a record of all filled slots in the conversation so far - for now, just use current_slots['inform_slots'] which is a dictionary of the already filled-in slots

        Returns:
        filled_in_slots             --  A dictionary of form {slot1:value1, slot2:value2} for each sloti in inform_slots_to_be_filled
        """

        kb_results = self.get_db_results(current_slots)
        #if dialog_config.auto_suggest == 1:
        #    print 'Number of entries in KB satisfying current constraints: ', len(kb_results)

        filled_in_slots = {}
        if const.TASK_COMPLETE_SLOT in inform_slots_to_be_filled.keys():
            filled_in_slots.update(current_slots[const.INFORM_SLOTS])

        for slot in inform_slots_to_be_filled.keys():
            if slot == 'numberofpeople':
                if slot in current_slots[const.INFORM_SLOTS].keys():
                    filled_in_slots[slot] = current_slots[const.INFORM_SLOTS][slot]
                elif slot in inform_slots_to_be_filled.keys():
                    filled_in_slots[slot] = inform_slots_to_be_filled[slot]
                continue

            if slot == 'ticket' or slot == const.TASK_COMPLETE_SLOT:
                filled_in_slots[slot] = const.TICKET_AVAILABLE if len(kb_results)>0 else const.NO_MATCH
                continue

            if slot == const.THANKS: continue

            ####################################################################
            #   Grab the value for the slot with the highest count and fill it
            ####################################################################
            values_dict = self._count_slot_values(slot, kb_results)

            values_counts = [(v, values_dict[v]) for v in values_dict.keys()]
            if len(values_counts) > 0:
                if inform_slots_to_be_filled[slot] == "PLACEHOLDER":
                    filled_in_slots[slot] = sorted(values_counts, key = lambda x: x[1], reverse=True)[0][0] # something wrong here
                else:
                    filled_in_slots[slot] = inform_slots_to_be_filled[slot]
            else:
                filled_in_slots[slot] = const.NO_MATCH #"NO VALUE MATCHES SNAFU!!!"

        return filled_in_slots

    def _count_slot_values(self, key, db_subdict):
        """
        Return a dict of the different values and occurrences of each, given a key, from a sub-dict of database

        Parameters:
            key (string): The key to be counted
            db_subdict (dict): A sub-dict of the database

        Returns:
            dict: The values and their occurrences given the key
        """

        slot_values = defaultdict(int)  # init to 0
        for id in db_subdict.keys():
            current_option_dict = db_subdict[id]
            # If there is a match
            if key in current_option_dict.keys():
                slot_value = current_option_dict[key]
                # This will add 1 to 0 if this is the first time this value has been encountered, or it will add 1
                # to whatever was already in there
                slot_values[slot_value] += 1
        return slot_values

    def get_db_results(self, constraints):
        """
        Get all items in the database that fit the current constraints.

        Looks at each item in the database and if its slots contain all constraints and their values match then the item
        is added to the return dict.

        Parameters:
            constraints (dict): The current informs

        Returns:
            dict: The available items in the database
        """

        # Filter non-queryable items and keys with the value 'anything' since those are
        # inconsequential to the constraints
        new_constraints = {k: v for k, v in constraints.items() if k not in self.no_query and v is not const.ANYTHING}

        inform_items = frozenset(new_constraints.items())
        cache_return = self.cached_db[inform_items]

        if cache_return is None:
            # If it is none then no matches fit with the constraints so return an empty dict
            return {}
        # if it isnt empty then return what it is
        if cache_return:
            return cache_return
        # else continue on

        available_options = {}
        for id in self.database.keys():
            current_option_dict = self.database[id]
            # First check if that database item actually contains the inform keys
            # Note: this assumes that if a constraint is not found in the db item then that item is not a match
            if len(set(new_constraints.keys()) - set(self.database[id].keys())) == 0:
                match = True
                # Now check all the constraint values against the db values and if there is a mismatch don't store
                for k, v in new_constraints.items():
                    if str(v).lower() != str(current_option_dict[k]).lower():
                        match = False
                if match:
                    # Update cache
                    self.cached_db[inform_items].update({id: current_option_dict})
                    available_options.update({id: current_option_dict})

        # if nothing available then set the set of constraint items to none in cache
        if not available_options:
            self.cached_db[inform_items] = None

        return available_options

    def get_db_results_for_slots(self, current_informs):
        """
        Counts occurrences of each current inform slot (key and value) in the database items.

        For each item in the database and each current inform slot if that slot is in the database item (matches key
        and value) then increment the count for that key by 1.

        Parameters:
            current_informs (dict): The current informs/constraints

        Returns:
            dict: Each key in current_informs with the count of the number of matches for that key
        """

        # The items (key, value) of the current informs are used as a key to the cached_db_slot
        inform_items = frozenset(current_informs.items())
        # A dict of the inform keys and their counts as stored (or not stored) in the cached_db_slot
        cache_return = self.cached_db_slot[inform_items]

        if cache_return:
            return cache_return

        # If it made it down here then a new query was made and it must add it to cached_db_slot and return it
        # Init all key values with 0
        db_results = {key: 0 for key in current_informs.keys()}
        db_results[const.KB_MATCHING_ALL_CONSTRAINTS] = 0

        for id in self.database.keys():
            all_slots_match = True
            for CI_key, CI_value in current_informs.items():
                # Skip if a no query item and all_slots_match stays true
                if CI_key in self.no_query:
                    continue
                # If anything all_slots_match stays true AND the specific key slot gets a +1
                if CI_value == const.ANYTHING:
                    db_results[CI_key] += 1
                    continue
                if CI_key in self.database[id].keys():
                    if CI_value.lower() == self.database[id][CI_key].lower():
                        db_results[CI_key] += 1
                    else:
                        all_slots_match = False
                else:
                    all_slots_match = False
            if all_slots_match:
                db_results[const.KB_MATCHING_ALL_CONSTRAINTS] += 1

        # update cache (set the empty dict)
        self.cached_db_slot[inform_items].update(db_results)
        assert self.cached_db_slot[inform_items] == db_results
        return db_results

    def suggest_slot_values(self, request_slots, current_slots):
        """ Return the suggest slot values """

        avail_kb_results = self.get_db_results(current_slots)


        return_suggest_slot_vals = {}
        for slot in request_slots.keys():
            avail_values_dict = self._count_slot_values(slot, avail_kb_results)
            values_counts = [(v, avail_values_dict[v]) for v in avail_values_dict.keys()]

            if len(values_counts) > 0:
                return_suggest_slot_vals[slot] = []
                sorted_dict = sorted(values_counts, key = lambda x: -x[1])
                for k in sorted_dict: return_suggest_slot_vals[slot].append(k[0])
            else:
                return_suggest_slot_vals[slot] = []

        return return_suggest_slot_vals
