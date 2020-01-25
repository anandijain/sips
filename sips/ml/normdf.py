import pandas as pd


def to_normed(df: pd.DataFrame, str_cols=["Game_id", "A_team", "H_team"]):
    strs = df[str_cols]
    df = df.drop(strs, axis=1)
    norm_df = (df - df.min()) / (df.max() - df.min())
    df = pd.concat([strs, norm_df], axis=1)
    return df


def norm_testset(
    test: pd.DataFrame, train: pd.DataFrame, str_cols=["Game_id", "A_team", "H_team"]
):
    train = train.drop(str_cols, axis=1)

    infer_strs = test[str_cols]  # to reattach post norm

    test = test.drop(infer_strs, axis=1)

    test = (test - train.min()) / (train.max() - train.min())

    normed_df = pd.concat([infer_strs, test], axis=1)
    print(normed_df)
    return normed_df
