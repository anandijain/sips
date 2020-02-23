import pandas as pd
import torch 

from sips.ml import normdf
from sips.ml import train
from sips.ml import prep
from sips.ml import data_loaders as dls

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


X_COLS = ['G',
          'MP',
          'PER',
          'TS%',
          '3PAr',
          'FTr',
          'ORB%',
          'DRB%',
          'TRB%',
          'AST%',
          'STL%',
          'BLK%',
          'TOV%',
          'USG%',
          'OWS',
          'DWS',
          'WS',
          'WS/48',
          'OBPM',
          'DBPM',
          'BPM',
          'VORP',
          'Salary',
          'Player_id',
          'NBA',
          'TOT']


Y_COLS = ['Salary']


def prep_sets():
    fn = '/home/sippycups/absa/sips/data/nba/player_career_stats_w_salaries2.csv'

    tr, te = prep.fn_to_tr_te(fn, 'Player_id', X_COLS, Y_COLS, norm_y=False)
    print(tr)
    print(te)

    tr_y = tr.Salary
    tr_x = tr.drop('Salary', axis=1)
    
    te_y = te.Salary
    te_x = te.drop('Salary', axis=1)
    
    dataset = dls.Salaryset(tr_x, tr_y)
    print(dataset[0])
    
    test_dataset = dls.Salaryset(te_x, te_y)
    return dataset, test_dataset

def train_sals(n:int=150):
    train_set, test_set = prep_sets()
    d = prep.prep_loader(train_set, test_set, "sals", device, batch_size=64, classify=False, shuffle=True)
    print(d)
    train.train(d, 'sals', n, verbose=False)


if __name__ == "__main__":
    train_sals()
