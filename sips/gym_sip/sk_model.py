#!/usr/bin/python3
import warnings
import pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import tree

import h

DATA_FOLDER = 'data/'

FILE_NAMES = ['2016reg.csv', '2017reg.csv', '2018reg.csv', '2019reg.csv']
TEST_FILES = ['2016playoffs.csv', '2017playoffs.csv', '2018playoffs.csv']

FEATURES = ['elo1_pre', 'elo2_pre', 'h_Home_Scorerol', 'h_Away_Scorerol', 'Home_Scorerol', 'Away_Scorerol']
TRAIN_COL = 'winner'

HOME_PREFIX = 'h_'
COL_TO_GET = 'ML'


def get_dfs(fns):
	dfs = []
	for fn in fns:
		fn = DATA_FOLDER + fn
		df = pd.read_csv(fn)
		df = df.dropna()
		df = df.drop_duplicates()
		dfs.append(df)
	return dfs


class Run:
	def __init__(self):
		dfs = get_dfs(FILE_NAMES)
		df = pd.concat(dfs, sort=True)

		x = df[FEATURES]
		y = df[TRAIN_COL]

		param_dist = {'n_estimators': np.arange(1, 50),
		              'learning_rate': np.arange(1, 10)/10,
		              'max_depth': np.arange(1, 3),
		              'random_state': np.arange(0, 10)}

		clfgtb = GradientBoostingClassifier()
		clfgtb = RandomizedSearchCV(clfgtb, param_dist,  cv=35)

		self.clfgtb = clfgtb.fit(x, y)

		test_dfs = get_dfs(TEST_FILES)
		test_df = pd.concat(test_dfs, sort=True)

		x_test = test_df[FEATURES]
		y_test = test_df[TRAIN_COL]

		print(str(clfgtb.score(x_test, y_test)) + ' gdpercent on first playoffs')

		probsgd = clfgtb.predict_proba(x_test)
		probsgd = probsgd.tolist()

		len_probsgd = len(probsgd)

		winners = test_df['winner']
		winners = list(winners)

		h_lines = test_df['h_ML']
		h_lines = list(h_lines)

		a_lines = test_df['ML']
		a_lines = list(a_lines)

		total = 0
		abets = []
		hbets = []
		allbets = []
		for i in range(len_probsgd):

			away_winprob = probsgd[i][0]
			home_winprob = probsgd[i][1]

			winner = winners[i]
			h_line = h_lines[i]
			a_line = a_lines[i]
			evhome = home_winprob * h.calc._eq(h_line) - away_winprob
			evaway = away_winprob * h.calc._eq(a_line) - home_winprob

			if winner == 'H':
				roi_home = h.calc._eq(h_line)
				roi_away = -1

			if winner == 'A':
				roi_home = -1
				roi_away = h.calc._eq(a_line)


			if evaway > 0:
				a_bets = [away_winprob, a_line, evaway, roi_away, winner]
				abets.append(a_bets)
				allbets.append(a_bets)

			if evhome > 0:
				h_bets = [home_winprob, h_line, evhome, roi_home, winner]
				hbets.append(h_bets)
				allbets.append(h_bets)


		all_df = pd.DataFrame(allbets, columns = ['winprob', 'line', 'ev', 'roi', 'winner'])
		home_df = pd.DataFrame(hbets, columns = ['home_winprob', 'h_line', 'evhome', 'roi_home', 'winner'])
		away_df = pd.DataFrame(abets, columns = ['away_winprob', 'a_line', 'evaway', 'roi_away', 'winner'])



		total_roi = all_df['roi'].sum()

a = Run()
g1 = [[1643, 1533, 116.5, 119, 106.5, 118.5]]

pred1 = a.clfgtb.predict(g1)

print(pred1)
