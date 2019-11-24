import tensorflow as tf

import sips.macros.bov as bm

import sips.h.helpers as h
import sips.h.serialize as s
import sips.h.attach as a

import sips.ml.tf_loaders as tfl
import sips.ml.tf_utils as tfu
import sips.ml.tf_lstm as lstm


def get_wl_datasets():
    dfs = h.get_dfs()
    all_dfs_w_labels = []
    for df in dfs:
        if df is not None:
            df.drop(["a_ou", "h_ou"], axis=1, inplace=True)
            # print(df)
            with_labels = a.wins(df)
            all_dfs_w_labels.append(with_labels)

    sXs, sYs = s.serialize_dfs(
        all_dfs_w_labels,
        in_cols=bm.TO_SERIALIZE,
        label_cols=["a_win", "h_win"],
        drop_labs=True,
    )

    datasets = tfl.serialized_to_datasets(sXs, sYs, history_size=30, pred_size=1)
    return datasets


def main():
    all_datasets = get_wl_datasets()
    datasets, test_datasets = h.train_test_split_list(all_datasets)
    x, y = tfu.get_example(datasets)

    loss_fxn = tf.losses.MeanAbsoluteError()
    optimizer = tf.keras.optimizers.RMSprop(clipvalue=1.0)
    model = lstm.make_model_functional(x.shape[-2:], tf.size(y[0]))
    return datasets


if __name__ == "__main__":
    datasets = main()
    print(datasets[0].element_spec)
