import lines as ll
import matplotlib.pyplot as plt 
import threading
import time

class Scraper:
    def __init__(self, file_name, header=0, game_type=1, run=1):
        self.fn = file_name
        self.gt = game_type
        self.header = header
        self.sip = ll.Sippy(self.fn, header, self.gt)
        self.sip.step()
        print('num_games: ' + str(len(self.sip.games)))
        if len(self.sip.games) > 0:
            self.game = self.sip.games[0]

        if run:
            self.run()

    def run(self):
        while True:
            self.sip.step()
            # for game in self.sip.games:
            #     action = sipqn.choose_action()

    def game_states():
        pass

