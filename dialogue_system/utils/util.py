
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
