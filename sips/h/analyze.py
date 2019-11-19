import numpy as np

from sips.h import helpers as h
from sips.macros import bov as bm
import sips.h.serialize as s


def classify_transition(prev_mls, cur_mls):
    """
    uses the propositions described in directional_transitions() to return a numpy array
    with the class of transition corresponding to the input moneylines
    """

    mls = [s.row_ml(ml) for ml in prev_mls + cur_mls]

    a_prev = mls[0]
    h_prev = mls[1]
    a_cur = mls[2]
    h_cur = mls[3]

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


def game_transitions(game, verbose=False):
    """
    given a dataframe of live lines for a single game,
    returns a list of classifications for the line movement
    """
    transition_classes = []
    teams_dict, statuses_dict = h.dicts_for_one_hotting()

    prev = [None, None]

    for i, row in game.iterrows():
        cur = list(row[["a_ml", "h_ml"]])
        transition_class = classify_transition(prev, cur)
        transition_classes.append(transition_class)
        prev = cur

    if verbose:
        strings = {i: s for i, s in enumerate(bm.TRANSITION_CLASS_STRINGS)}

        for i, t in enumerate(transition_classes):
            class_num = np.argmax(t)
            print(f"{i}: {strings[class_num]}")

    return transition_classes
