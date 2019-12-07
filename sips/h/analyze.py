"""
directional moneyline transition classes

"""
import numpy as np


def classify_transition(prev_mls, cur_mls):
    """
    uses the propositions described in directional_transitions() to return a numpy array
    with the class of transition corresponding to the input moneylines

    """
    a_prev, h_prev = prev_mls
    a_cur, h_cur = cur_mls

    propositions = directional_transitions(a_prev, a_cur, h_prev, h_cur)
    ret = np.zeros(len(propositions))

    for i, phi in enumerate(propositions):
        if phi:
            ret[i] = 1
            break

    return ret


def directional_transitions(a1, a2, h1, h2):
    """
    classification of the movement of lines where -1 is closed
    """
    # how to metaprogram the enumeration of combinations given binary relations
    propositions = [
        # opening actions
        ((a1 == -1 and a2 != -1) and (h1 == -1 and h2 != -1)),
        ((a1 == -1 and a2 != -1) and (h1 < h2)),
        ((a1 == -1 and a2 != -1) and (h1 > h2)),
        ((a1 < a2) and (h1 == -1 and h2 != -1)),
        ((a1 > a2) and (h1 == -1 and h2 != -1)),
        # closing actions
        ((a1 and a2 == -1) and (h1 and h2 == -1)),
        ((a1 and a2 == -1) and (h1 < h2)),
        ((a1 and a2 == -1) and (h1 > h2)),
        ((a1 < a2) and (h1 and h2 == -1)),
        ((a1 > a2) and (h1 and h2 == -1)),
        # directionals
        (a1 == a2 and h1 == h2),
        (a1 < a2 and h1 == h2),
        (a1 > a2 and h1 == h2),
        (a1 == a2 and h1 < h2),
        (a1 == a2 and h1 > h2),
        (a1 < a2 and h1 < h2),
        (a1 > a2 and h1 > h2),
        (a1 < a2 and h1 > h2),
        (a1 > a2 and h1 < h2),
    ]
    return propositions
