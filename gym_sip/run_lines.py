import lines as ll
import matplotlib.pyplot as plt 
import threading
import time

class Start:
    def __init__(self, fn='nba2', header=0, game_type=1, run=1):
        self.fn = fn
        self.gt = game_type
        self.header = header
        self.sip = ll.Sippy(fn=self.fn, header=self.header, league=self.gt)
        self.sip.step()
        print('num_games: ' + str(len(self.sip.games)))
        if len(self.sip.games) > 0:
            self.game = self.sip.games[0]
        if run != 0:
            self.run()
        # self.game.info()

    def run(self):
        self.sip.run()
