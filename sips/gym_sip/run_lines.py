import time

import sips.gym_sip.lines as ll

class Start:
    """
    fn - String, csv file path to './sips/gym_sip/data/static/' + fn + '.csv'

    game_type - String in ['all', 'nba', 'college basketball', 'mlb', 'esports',
                'football', 'tennis', 'volleyball', 'hockey']

    run - Bool, to automatically run

    verbosity - Bool, extra print statements
    """

    def __init__(self, fn='all', game_type='all', run=True, verbosity=False):
        self.fn = fn
        self.gt = game_type

        self.sip = ll.Sippy(fn=self.fn, league=self.gt, verbosity=verbosity)
        self.sip.step()

        if len(self.sip.games) > 0:
            self.game = self.sip.games[0]
        if run:
            self.run()

    def run(self):
        self.sip.run()

    def __repr__(self):
        print(self.sip)


if __name__ == "__main__":
    sip = Start(fn='all', game_type='all')
