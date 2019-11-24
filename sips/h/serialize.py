import pandas as pd
import numpy as np

import tensorflow as tf

from sips.macros import bov as bm
from sips.h import hot
import sips.h.helpers as h


def serialize_row(row, hot_maps=None, to_numpy=True, include_teams=False):
    """

    """
    if isinstance(row, pd.core.series.Series):
        pass
    elif isinstance(row, list):
        row = pd.DataFrame(row, columns=bm.LINE_COLUMNS)

    ret = serialize_df(row, hot_maps=hot_maps)

    if to_numpy:
        ret = np.array(ret, dtype=np.float32)

    return ret


def serialize_dfs(
    dfs,
    in_cols,
    label_cols,
    replace_dict=None,
    hot_maps=None,
    to_numpy=True,
    norm=True,
    dropna=True,
    dont_hot=False,
    drop_labs=True
):
    """
    label_cols is a subset of incols
    """
    sXs = []
    sYs = []
    if dont_hot:
        hot_maps = None
    else:
        if not hot_maps:
            hot_maps = hot.all_hot_maps(output="dict")

    if not replace_dict:
        replace_dict = {"None": np.nan, "EVEN": 100}  # hacky

    for df in dfs:
        try:
            subset = df[in_cols].copy()
        except KeyError:
            continue
        except TypeError:
            continue

        sdf = serialize_df(
            df, replace_dict=replace_dict, hot_maps=hot_maps, dropna=dropna
        )
        # typed_df = sdf.astype(np.float32)

        y = sdf[label_cols].copy()
        X = sdf.copy()
        
        if drop_labs:
            X = sdf.drop(label_cols, axis=1)

        if to_numpy:
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            if norm:
                X = tf.keras.utils.normalize(X)
        sXs.append(X)
        sYs.append(y)

    return sXs, sYs


def serialize_df(df, replace_dict=None, hot_maps=None, dropna=True):
    """
    built to take in dataframe from running lines.py
    df: pd.DataFrame
    replace_dict: dict replaces values like 'None', and 'E
    """

    if replace_dict:
        df.replace(replace_dict, inplace=True)

    if hot_maps:
        df = hot.hot(df, hot_maps=hot_maps)

    if dropna:
        df.dropna(inplace=True)

    ret = df
    return ret


def row_ml(ml):
    """
    given a list of unparsed moneylines (eg can be 'EVEN' and None)
    edit the values such that 'EVEN' -> 100 and None -> -1
    typical order of list is [a0, h0, a1, h1]
    """
    if ml == "EVEN":
        ret = 100
    elif ml == None:
        ret = -1
    else:
        try:
            ret = float(ml)
        except:
            ret = -1
    return ret


def teams(row, teams_dict):
    """
    row is one of:
        row of type list (with schema specified in serialize row)
        pandas row
    teams_dict is:
        team_name: hotted vector
    """
    ret = []

    if isinstance(row, pd.core.series.Series):
        a_team = row.a_team
        h_team = row.h_team
    else:
        a_team, h_team = row[2:4]

    for t in [a_team, h_team]:
        hot_team = teams_dict[t]
        ret += hot_team

    return ret


def test_sdfs():

    dfs = h.get_dfs()
    cols = bm.TO_SERIALIZE
    maps = hot.all_hot_maps()

    numbers = serialize_dfs(
        dfs, in_cols=cols, label_cols=["a_ml", "h_ml"], hot_maps=maps
    )

    print(numbers)
    return numbers


if __name__ == "__main__":
    test_sdfs()
    # pd.read_csv('')
