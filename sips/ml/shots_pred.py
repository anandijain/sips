import pandas as pd


from sips.ml import normdf
from sips.ml import train
from sips.ml import data_loaders as dls



def shots_pred_train():

    df = pd.read_csv('1000_shots.csv')
    x = ['game_id', 'qtr_x', 'x_pos', 'y_pos', 'tot_sec', 'home']
    y = ['a_pts', 'h_pts']

    df = df[x + y]
    tr, te = normdf.split_norm(df)

    trset = dls.Shotset(tr)
    teset = dls.Shotset(te)

    x, y = trset[0].values()
    print(f'x{x}')
    print(f'x{x.shape[0]}')

    print(f'y{y}')
    print(f'y{y.shape[0]}')

    d = train.prep_loader(trset, teset)
    train.train(d, 'shots_pred.pth')

if __name__ == "__main__":

    d = shots_pred_train()
    print(d)