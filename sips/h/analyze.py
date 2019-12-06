import pandas as pd
import numpy as np

from sips.h import serialize as s
from sips.h import calc
from sips.h import helpers

def summary(df: pd.DataFrame, output='list'):
    """
    takes a dataframe of lines and returns a single row dataframe or list
    per team:
        num_changes
        total_change
        min_change
        max_change
        25, 50, 75% changes
        std, var changes
        avg change
        num closures

        min, max, quartiles, std general 
    """
    cols = ['a_ml', 'h_ml']
    # for col in df[cols]:
    a_ml = df.a_ml
    h_ml = df.h_ml

    a_eqs = list(map(calc.eq, a_ml))
    h_eqs = list(map(calc.eq, h_ml))

    a_deltas = calc.deltas(a_eqs)
    h_deltas = calc.deltas(h_eqs)

    hdf = pd.Series(a_deltas, name='h_deltas')
    adf = pd.Series(h_deltas, name='a_deltas')
    ret = pd.concat([adf, hdf], axis=1)
    return ret

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

if __name__ == "__main__":
    dfs = helpers.get_dfs()
    sdfs = s.serialize_dfs(dfs, to_numpy=False, astype=np.float32, norm=False)
    sdf = sdfs[0]
    df = summary(sdf)
    print(df)
    print(df.describe())
