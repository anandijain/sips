import pandas as pd


from sips.ml import normdf
from sips.ml import train
from sips.ml import data_loaders as dls

MODEL_NAME = 'shots'
BATCH_SIZE = 256

X_COLS = ['qtr_x', 'x_pos', 'y_pos',
     'tot_sec', 'home', 'a_pts', 'h_pts']

Y_COLS = ['made', 'missed']

def shots_df():
    df = pd.read_csv('1000_shots.csv')
    made_missed = pd.get_dummies(df.shot_made)
    print(made_missed)
    df[['made', 'missed']] = made_missed
    df.drop('shot_made', axis=1, inplace=True)
    cols = ['game_id'] + X_COLS + Y_COLS
    df = df[cols]
    df = df.sample(frac=1).reset_index(drop=True)

    return df

def shots_prep(model_name):

    df = shots_df()
    tr, te = normdf.split_norm(df)

    trset = dls.Shotset(tr, feat_cols=X_COLS, lab_cols=Y_COLS)
    teset = dls.Shotset(te, feat_cols=X_COLS, lab_cols=Y_COLS)

    x, y = trset[0].values()

    print(f'x{x}')
    print(f'x{x.shape[0]}')

    print(f'y{y}')
    print(f'y{y.shape[0]}')

    d = train.prep_loader(trset, teset, model_name, classify=True, batch_size=BATCH_SIZE)
    return d

def shots_pred_train():

    d = shots_prep(MODEL_NAME)
    train.train(d, 'shots_pred_classify.pth')

if __name__ == "__main__":

    shots_pred_train()
