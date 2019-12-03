import json

def action_to_string(action):
    action = json.loads(action)

    if action:
        intent = action['intent']
        request_slots = action['request_slots'].keys()
        inform_slots = action['inform_slots']

        inform_str = ''
        for inform in inform_slots:
            inform_str += f"; '{inform}': {inform_slots[inform]}"

        action_str = f"{intent}({'; '.join(request_slots)}{inform_str})"

        return action_str

    return action

def goal_to_string(goal):
    goal = json.loads(goal)

    if goal:
        request_slots = list(goal['request_slots'].keys())
        inform_slots = goal['inform_slots']

        inform_str = ''
        for inform in inform_slots:
            inform_str += f"{inform}: {inform_slots[inform]}; "

        goal_str = f"Goal:\nrequest: {request_slots}\ninform: { {inform_str[:-2]} }"

        return goal_str

    return goal

def informs_to_string(current_informs, db_count):
    current_informs = json.loads(current_informs)
    db_count = json.loads(db_count)

    if current_informs and db_count:

        count = db_count['matching_all_constraints']

        inform_str = '['
        for inform in current_informs:
            inform_str += f"{inform}: {current_informs[inform]}; "

        inform_str += ']'

        informs_string = f"Entidades informadas:\n{inform_str}\n\nNum. de ingressos:\n{count}"

        return informs_string

    return ""

def db_to_string(db_results):
    db_results = json.loads(db_results)
    db_results_str = []

    if db_results:

        for result in db_results:
            result_str = ''

            for slot in result:
                result_str += f"'{slot}': {result[slot]}; "

            db_results_str.append(result_str)

    return db_results_str
