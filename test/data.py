import sips.h.helpers as h
import sips.macros.macros as m


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
