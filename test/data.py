import time

import sips

from sips.h import attach
from sips.h import hot
from sips.h import helpers as h
from sips.h import serialize as s
from sips.h import attach

from sips.macros import macros as m
from sips.macros import bov as bm


def test_get_and_window():

    columns = ["a_pts", "h_pts", "quarter", "secs"]
    dfs = h.get_dfs(m.PARENT_DIR + "data/lines/lines/")

    sets = [
        h.window_multivariate(
            df.values, df[columns].values, history_size=10, target_size=10
        )
        for df in dfs
    ]
    first = sets[0]
    X, y = first
    print(f"X.shape : {X.shape}")
    print(f"y.shape : {y.shape}")
    return X, y


def test_attach_wins():
    # get dataframes and attach the win/losses
    dfs = h.get_dfs()
    w_wins = attach.wins(dfs)
    return w_wins


def test_get_filter_and_serialize():
    dfs = h.get_dfs()
    data = s.serialize_dfs(dfs)
    sdfs = s.serialize_dfs(
        dfs, label_cols=["a_pts", "h_pts", "a_ml", "h_ml"], to_numpy=False
    )

    zipped = list(zip(sdfs[0], sdfs[1]))
    print(len(zipped))
    print(zipped[0])
    return data, sdfs


def df_filtering_commutative_time_delta():
    """ 
    Tests whether applying a min game length then filtering for wins
    is faster than filtering the other way around.

    Both are slow and the filter needs to be done in one iteration of the dfs.

    """
    all_dfs = h.get_dfs()
    num_dfs_initial = len(all_dfs)

    start1 = time.time()
    apply_then_filter = h.apply_min_then_filter(all_dfs, verbose=True)
    end1 = time.time()

    start2 = time.time()
    filter_then_apply_min = h.filter_then_apply_min(all_dfs, verbose=True)
    end2 = time.time()

    delta1 = end1 - start1
    delta2 = end2 - start2
    if len(filter_then_apply_min) > 0:
        print(f"df: {filter_then_apply_min[0]}")
        print(f"df: {filter_then_apply_min[0].status}")

    print(f"delta 1: {delta1}")
    print(f"delta 2: {delta2}")
    return delta1, delta2


def test_sdfs():
    dfs = h.get_dfs()
    cols = bm.TO_SERIALIZE
    maps = hot.all_hot_maps()

    numbers = s.serialize_dfs(
        dfs, in_cols=None, label_cols=None, hot_maps=maps, to_numpy=False
    )

    print(numbers)
    return numbers


def test_heat():
    dfs = h.get_dfs()
    df = dfs[0]

    hot_maps = hot.all_hot_maps(output="dict")
    hotted = hot.hot(df, hot_maps=hot_maps)
    return hotted


if __name__ == "__main__":
    X, y = test_get_and_window()
    w_wins = test_attach_wins()
    data, sdfs = test_get_filter_and_serialize()
    d1, d2 = df_filtering_commutative_time_delta()
    numbers = test_sdfs()
    hotted = test_heat()
