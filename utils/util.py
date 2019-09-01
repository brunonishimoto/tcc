import json
from setup_logger import loggers

def convert_list_to_dict(lst):
    """
    Convert list to dict where the keys are the list elements, and the values are the indices of the elements
    in the list.

    Parameters:
        lst (list)

    Returns:
        dict
    """

    if len(lst) > len(set(lst)):
        raise ValueError('List must be unique!')
    return {k: v for v, k in enumerate(lst)}


def remove_empty_slots(dic):
    """
    Removes all items with values of '' (ie values of empty string).

    Parameters:
        dic (dict)
    """

    for id in list(dic.keys()):
        for key in list(dic[id].keys()):
            if dic[id][key] == '':
                dic[id].pop(key)

def save_json_file(path, data):
    """Save a json file."""
    try:
        json.dump(data, open(path, "w"), indent=2)
        print(f'saved data in {path}')
    except Exception as e:
        print(f'Error: Writing model fails: {path}')
        print(e)

def log(names, msg):
    for name in names:
        loggers[name].info(msg)
