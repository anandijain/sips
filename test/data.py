import time

from sips.h import helpers as h
from sips.h import attach
from sips.h import serialize as s
from sips.macros import macros as m


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
    dfs = h.get_dfs()
    w_wins = [attach.wins(df) for df in dfs]
    return w_wins


def test_get_filter_and_serialize():
    dfs = h.get_dfs()
    data = s.serialize_dfs(dfs)
    sdfs = s.serialize_dfs(
        dfs, label_cols=['a_pts', 'h_pts', 'a_ml', 'h_ml'], to_numpy=False)

    zipped = list(zip(sdfs[0], sdfs[1]))
    print(len(zipped))
    print(zipped[0])
    return data, sdfs


def commutative_time_delta():
    all_dfs = h.get_dfs()

    start1 = time.time()
    apply_then_filter = h.apply_min_then_filter(all_dfs)
    end1 = time.time()

    start2 = time.time()
    filter_then_apply_min = h.filter_then_apply_min(all_dfs)
    end2 = time.time()

    print(f'delta 1: {end1 - start1}')
    print(f'delta 2: {end2 - start2}')
