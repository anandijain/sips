import matplotlib.pyplot as plt
from sips.h import helpers as h


class Viz:
    def __init__(self, fn='static/nba2'):
        self.fn = fn
        self.games = h.get_games(self.fn)

    def graph_game(self, index):
        # could be game_id for dictionary, assuming list tho
        game = self.games[index]
        data = game[['cur_time', 'a_ml', 'h_ml']]
        print(data.head())

        plt.plot('cur_time', 'a_ml', data=game, marker='o',
                 markerfacecolor='blue', markersize=.5,
                 color='skyblue', linewidth=.5)
        plt.plot('cur_time', 'h_ml', data=game, marker='x',
                 markerfacecolor='red', markersize=.5,
                 color='red', linewidth=.5)
        plt.legend()
        plt.show()
