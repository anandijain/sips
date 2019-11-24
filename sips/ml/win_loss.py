import sips.h.helpers as h
import sips.h.serialize as s
import sips.h.attach as a
import sips.macros.bov as bm
import sips.ml.tf_loaders as tfl


def get_wl_datasets():
    dfs = h.get_dfs()
    all_dfs_w_labels = []
    for df in dfs:
        if df is not None:
            df.drop(['a_ou', 'h_ou'], axis=1, inplace=True)
            # print(df)
            with_labels = a.wins(df)
            all_dfs_w_labels.append(with_labels)

    sXs, sYs = s.serialize_dfs(all_dfs_w_labels, in_cols=bm.TO_SERIALIZE,
                               label_cols=['a_win', 'h_win'], drop_labs=True)

    datasets = tfl.serialized_to_datasets(
        sXs, sYs, history_size=30, pred_size=30)
    return datasets


def main():
    datasets = get_wl_datasets()
    return datasets


if __name__ == "__main__":
    datasets = main()
    print(datasets)
