import pandas as pd
import numpy as np

from sips.ml import prep

if __name__ == "__main__":
    fn = '/home/sippycups/absa/sips/sips/ml/players_and_teams.csv'
    tr, te = prep.fn_to_tr_te(fn, by='Game_id')
    print(tr)
    print(te)