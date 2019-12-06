import pandas as pd
import numpy as np

from sips.h import serialize as s
from sips.h import calc
from sips.h import helpers


def summaries(dfs: list, columns=['a_ml', 'h_ml', 'a_hcap', 'h_hcap'], verbose=False) -> pd.DataFrame:
    rows = []
    for sdf in sdfs:
        row = summary(sdf, columns=columns)
        rows.append(row)

    summaries = pd.DataFrame(rows, columns=columns)
    if verbose:
        print(summaries)
        print(summaries.describe())

    return summaries


def summary(df: pd.DataFrame, columns=['a_ml', 'h_ml', 'a_hcap', 'h_hcap', 'a_ps', 'h_ps'], output="list"):
    """
    takes a dataframe of lines and returns a single row dataframe or list
    per team:
        num_changes
        total_change
        min_change
        max_change
        25, 50, 75% changes
        std changes
        avg change
        num closures

        min, max, quartiles, std general
    """
    subset = df[columns]
    desc = subset.describe()
    deltas = []

    for i, col in subset.iteritems():
        deltas.append(calc.deltas(col))

    deltas_df = pd.DataFrame(deltas)
    data = desc.values.tolist() + deltas_df.describe().values.tolist()
    flat_list = [item for sublist in data for item in sublist]

    return flat_list


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

    single_team_lines_summary_columns = [
        "num",
        "mean",
        "std",
        "min",
        "25",
        "50",
        "75",
        "max",
        "num_eq_ch",
        "num_ch",
        "eq_mean",
        "ch_mean",
        "eq_std",
        "ch_std",
        "eq_min",
        "ch_min",
        "eq_25",
        "ch_25",
        "eq_50",
        "ch_50",
        "eq_75",
        "ch_75",
        "eq_max",
        "ch_max",
    ]

    dfs = helpers.get_dfs()
    sdfs = s.serialize_dfs(dfs, to_numpy=False, astype=np.float32, norm=False)
    summs = summaries(sdfs)
    print(summs)