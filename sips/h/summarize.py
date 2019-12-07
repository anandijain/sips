"""

"""
import pandas as pd
import numpy as np

from sips.h import serialize as s
from sips.h import calc
from sips.h import helpers


def summaries(
    dfs: list, columns=["a_ml", "h_ml", "a_hcap", "h_hcap"], verbose=False
) -> pd.DataFrame:
    """
    given a list of serialized dataframes, an aggregate summary df is returned

    """
    rows = []
    summ_cols = get_summary_cols(columns)

    for sdf in dfs:
        row = summary(sdf, columns=columns, summ_cols=summ_cols, output="df")
        rows.append(row)

    summs = pd.concat(rows)
    summs.reindex()

    if verbose:
        print(summs)
        print(summs.describe())

    return summs


def summary(
    df: pd.DataFrame,
    columns=["a_ml", "h_ml", "a_hcap", "h_hcap", "a_ps", "h_ps"],
    summ_cols=None,
    output="list",
):
    """
    takes a dataframe of lines and returns a single row dataframe or list

    """
    subset = df[columns]
    desc = subset.describe()
    desc_data = desc.values.tolist()
    deltas = []

    for _, col in subset.iteritems():
        deltas.append(calc.deltas(col))

    deltas_df = pd.DataFrame(deltas).transpose()
    deltas_summ_data = deltas_df.describe().values.tolist()

    data = desc_data + deltas_summ_data
    ret = [item for sublist in data for item in sublist]

    if output == "df":
        if not summ_cols:
            summ_cols = get_summary_cols(columns)
        ret = pd.DataFrame([ret], columns=summ_cols)

    return ret


def get_summary_cols(cols_to_summarize: list, verbose=False):
    """
    constructs columns for summary df

    """
    single_team_lines_summary_columns = [
        "num",
        "mean",
        "std",
        "min",
        "25",
        "50",
        "75",
        "max",
        "num_ch",
        "ch_mean",
        "ch_std",
        "ch_min",
        "ch_25",
        "ch_50",
        "ch_75",
        "ch_max",
    ]  # 16 per stat
    cols = [
        stat + "_" + summary_stat
        for summary_stat in single_team_lines_summary_columns
        for stat in cols_to_summarize
    ]
    if verbose:
        print(cols)
    return cols


if __name__ == "__main__":
    cols_to_summarize = ["a_ml", "h_ml", "a_hcap", "h_hcap", "a_ps", "h_ps"]
    dfs = helpers.get_dfs()
    sdfs = s.serialize_dfs(dfs, to_numpy=False, astype=np.float32, norm=False)
    summs = summaries(sdfs, columns=cols_to_summarize, verbose=True)
