

def df_col_type_dict(df):
    # given a df, returns a dict where key is column name, value is dtype
    return dict(zip(list(df.columns), list(df.dtypes)))


def data_sample(dataset):
    for i in range(len(dataset)):
        sample = dataset[i]

        print(i, sample["x"], sample["y"])
        if i == 0:
            x_shape = sample["x"].shape
            y_shape = sample["y"].shape
            print(f"x_shape: {x_shape}")
            print(f"y_shape: {y_shape}")
            break


def nums_only(df):
    df = df.select_dtypes(exclude=["object"])
    df = df.apply(pd.to_numeric)
    return df
