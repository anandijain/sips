import os
import time
import json 

import sips.h.openers as io
from sips.lines.bov import bov
from sips.lines.bov.utils import bov_utils as utils

class Lines:
    '''
    sport (str):
        - used to construct the api urls

    wait: (positive number)
        - number of seconds to sleep between each request

    write_config: 
        - if true, data is only written if it is different than previous
        , sport='nfl', wait=5, start=True, write_new=False, verbose=False
    '''
    def __init__(self, config_path='./config/newl.json'):
        self.dir = '../data/lines/'
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

        with open(config_path) as config:
            self.config = json.load(config)
        # print(self.config)
        self.sports = self.config.get('sports')

        print(f'sports: {self.sports}')
        self.wait = self.config.get('wait')
        self.write_new = self.config.get('write').get('new_only')
        self.verbose = self.config.get('verbose')

        start = self.config['start']

        # dict of game files 
        self.files = {}

        self.prevs = None
        self.news = None

        if start:
            self.run()

    def step(self):
        self.news = bov.lines(self.sports)
        print(f'self.news: {self.news}')
        to_write = prepare_write(self.prevs, self.news)
        self.files = write_data(self.files, to_write)
        self.prevs = self.news
        time.sleep(self.wait)

    def run(self):
        try:
            self.prevs = bov.lines(self.sports)
            time.sleep(0.5)
            while True:
                self.step()
        except KeyboardInterrupt:
            print('interrupted')


def prepare_write(prevs, news):
    '''
    input: both are dictionaries of (game_id, event) 

    returns: dictionary of (game_id, row_data) of new data
    '''

    to_write = {}

    for k, v in news.items():
        if not prevs.get(k) or prevs.get(k) != v:
            to_write[k] = v
        else:
            continue
    return to_write


def init_file(fn):
    f = open(fn, 'a')
    io.write_list(f, utils.header())
    f.close()


def write_data(file_dict, data_dict):
    '''

    '''
    for k, v in data_dict.items():
        f = file_dict.get(k)
        fn = '../data/lines/' + str(k) + '.csv'
        if not f:
            file_dict[k] = fn
            if not os.path.isfile(fn):
                init_file(fn)

        f = open(fn, 'a')

        io.write_list(f, v)

        f.close()
    return file_dict


def update_check(prev, new):
    '''
    prev - list of old data
    new - list of new data
    returns - bool
    '''
    is_updated = True if prev == new else False
    return is_updated


if __name__ == '__main__':
    line = Lines('./config/newl.json')
