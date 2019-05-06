import mlbgame as m
import pandas as pd
import numpy as np 

def games_to_csv(year_range):
	# year range is a tuple. eg (2010, 2019)
	while year_range[0] <= year_range[1]:
		gs = m.games(year_range[0])
		to_csv(gs)  # TODO


def get_game():
	game_stats = mlbgame.team_stats('2016_08_02_nyamlb_nynmlb_1')
	return game_stats 

g = get_game()

