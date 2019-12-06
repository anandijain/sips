import tensorflow as tf
import numpy as np
import pandas as pd

from sips.h import serialize as s
from sips.h import helpers as h
from sips.h import attach
from sips.h import analyze
from sips.h import fileio as fio
from sips.h import hot

from sips.macros import tfm

from sips.macros import bov as bm
from sips.macros import macros as m
import sips.h.calc as c


def transition_datasets_from_folder(
    folder=tfm.READ_FROM,
    hot_maps=None,
    history_size=1,
    pred_size=1,
    batch_size=1,
    single_step=False,
):
    """

    """
    if not hot_maps:
        hot_maps = hot.all_hot_maps()

    dfs = h.get_dfs(folder)
    sXs = []
    sYs = []
    for df in dfs:
        df.drop(["a_ou", "h_ou"], axis=1, inplace=True)
        X, y = prep_game_dataset(df, hot_maps)
        sXs.append(X)
        sYs.append(y)

    datasets = serialized_to_datasets(
        sXs,
        sYs,
        history_size=history_size,
        pred_size=pred_size,
        batch_size=batch_size,
        single_step=single_step,
    )

    return datasets


def prep_game_dataset(df, hot_maps=None):
    """

    """
    sdf = s.serialize_df(df, hot_maps=hot_maps)
    transitions = attach.ml_transitions(df, attach=False)

    return np.array(sdf, dtype=np.float32), transitions


def prediction_data_from_folder(
    folder=tfm.READ_FROM,
    in_cols=None,
    label_cols=["a_ml", "h_ml"],
    batch_size=1,
    buffer_size=1,
    history_size=10,
    pred_size=2,
    step_size=1,
    norm=True,
    verbose=False,
):
    """

    """
    replace_data_map = {"None": np.nan, "EVEN": 100}
    hot_maps = hot.all_hot_maps(output="dict")
    dfs = h.get_dfs(folder)

    train_df_vals, train_df_labs = s.serialize_dfs(
        dfs,
        label_cols=label_cols,
        replace_dict=replace_data_map,
        hot_maps=hot_maps,
        norm=norm,
    )
    if verbose:
        print(f"train_df_vals: {train_df_vals}")
        print(f"train_df_labs: {train_df_labs}")

    datasets = serialized_to_datasets(
        train_df_vals,
        train_df_labs,
        history_size=history_size,
        pred_size=pred_size,
        batch_size=batch_size,
    )

    return datasets


def serialized_to_datasets(
    train_df_vals,
    train_df_labs,
    history_size=1,
    pred_size=1,
    batch_size=1,
    single_step=False,
):
    """

    """

    datasets = []
    for i in range(len(train_df_vals)):
        data = train_df_vals[i].astype(np.float32)
        length = len(data)
        if length < history_size + pred_size:
            continue
        targets = train_df_labs[i]
        X_windows, y_windows = h.window_multivariate(
            data,
            targets,
            history_size=history_size,
            target_size=pred_size,
            single_step=single_step,
        )

        x_dataset = tf.data.Dataset.from_tensor_slices(X_windows)
        y_dataset = tf.data.Dataset.from_tensor_slices(y_windows)
        prepped = tf.data.Dataset.zip((x_dataset, y_dataset)).batch(batch_size)
        datasets.append(prepped)

    datasets = list(filter(None, datasets))
    return datasets


# def test_datasets_from_dir():
#     """
#     attempts to read folder and convert to train/test tf datasets lists
#     """
#     datasets = datasets_from_dir(df_to_tf_dataset)
#     return datasets


if __name__ == "__main__":
    datasets = transition_datasets_from_folder()
    print(len(datasets))
