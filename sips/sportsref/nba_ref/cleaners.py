import pandas as pd


def four_factors(df):
    df.columns = df.iloc[0]
    df = df.drop(0)
    df.columns[0] = 'win'
    df.columns[1] = 'team'
    return df