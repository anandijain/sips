import pandas as pd


from sips.ml import normdf
from sips.ml import train
from sips.ml import data_loaders as dls

MODEL_NAME = 'shots_to_score'
BATCH_SIZE = 256
CLASSIFY = False
X_COLS = ['game_id', 'qtr_x', 'x_pos', 'y_pos',
          'tot_sec', 'home', 'a_pts', 'h_pts', 'shot_made']
Y_COLS = ['a_pts', 'h_pts']
# Y_COLS = ['made', 'missed']
# todo sequential model, given first 10 shots and made miss, predict w/l

def shots_pred_train():
    d = shots_prep()
    train.train(d, MODEL_NAME)


def shots_prep():

    tr, te = shots_tr_te()
    tr_y = tr[Y_COLS]
    tr.drop(Y_COLS, axis=1, inplace=True)
    tr_x = tr

    te_y = te[Y_COLS]
    te.drop(Y_COLS, axis=1, inplace=True)
    te_x = te

    trset = dls.Shotset(tr_x, tr_y)
    teset = dls.Shotset(te_x, te_y)

    x, y = trset[0].values()

    print(f'x{x}')
    print(f'x{x.shape[0]}')

    print(f'y{y}')
    print(f'y{y.shape[0]}')

    d = train.prep_loader(trset, teset, MODEL_NAME,
                          classify=CLASSIFY, batch_size=BATCH_SIZE)
    return d


def shots_tr_te():
    df = pd.read_csv('1000_shots.csv')
    df = df[X_COLS]
    df = df.sample(frac=1).reset_index(drop=True)
    tr, te = normdf.split_norm(df, y_cols=Y_COLS)
    tr = tr.reset_index(drop=True)
    te = te.reset_index(drop=True)
    return tr, te


def shot_classify(df):
    made_missed = pd.get_dummies(df.shot_made)
    df[['made', 'missed']] = made_missed
    df.drop('shot_made', axis=1, inplace=True)
    return df


if __name__ == "__main__":

    shots_pred_train()
