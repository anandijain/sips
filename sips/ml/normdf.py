import pandas as pd
import numpy as np

from sips.h import helpers


def to_normed(df: pd.DataFrame, str_cols=["game_id", "a_team", "h_team"]):
    if str_cols:
        strs = df[str_cols]
        df = df.drop(strs, axis=1)

    normed_df = (df - df.min()) / (df.max() - df.min())

    if str_cols:
        normed_df = pd.concat([strs, normed_df], axis=1)
    return normed_df


def norm_testset(
    test: pd.DataFrame, train: pd.DataFrame, str_cols=["game_id", "a_team", "h_team"]
):
    if str_cols:
        train = train.drop(str_cols, axis=1)
        infer_strs = test[str_cols]  # to reattach post norm
        test = test.drop(infer_strs, axis=1)

    test_normed = (test - train.min()) / (train.max() - train.min())

    if str_cols:
        test_normed = pd.concat([infer_strs, test], axis=1)

    return test_normed


def split_norm(df, by='game_id', y_cols=['a_pts', 'h_pts'], str_cols=None, train_frac=0.7):
    tr_df, te_df = helpers.split_by(df, by=by, train_frac=train_frac)
   
    tr_y = tr_df[y_cols]
    te_y = te_df[y_cols]
        
    tr_df.drop(y_cols, axis=1, inplace=True)
    te_df.drop(y_cols, axis=1, inplace=True)

    tr_df.drop(by, axis=1, inplace=True)
    te_df.drop(by, axis=1, inplace=True)

    te_normed = norm_testset(te_df, tr_df, str_cols=str_cols)
    tr_normed = to_normed(tr_df, str_cols=str_cols)

    tr_normed[y_cols] = tr_y
    te_normed[y_cols] = te_y

    return tr_normed, te_normed

def splitnorm(df, by='game_id', y_cols=['a_pts', 'h_pts'], str_cols=None, train_frac=0.7, norm_y=False):
    tr_df, te_df = helpers.split_by(df, by=by, train_frac=train_frac)
    tr_df.drop(by, axis=1, inplace=True)
    te_df.drop(by, axis=1, inplace=True)

    te_normed = norm_testset(te_df, tr_df, str_cols=str_cols)
    tr_normed = to_normed(tr_df, str_cols=str_cols)

    return tr_normed, te_normed
