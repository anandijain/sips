import os
import time
import json 

import sips.h.openers as io
from sips.lines.bov import bov
from sips.lines.bov.utils import bov_utils as utils
from sips.h import openers

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
    def __init__(self, config_path='./config/new_lines.json'):
        self.dir = '../data/lines/'

        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

        with open(config_path) as config:
            self.config = json.load(config)

        self.sports = self.config.get('sports')
        print(f'sports: {self.sports}')

        self.wait = self.config.get('wait')
        self.write_new = self.config.get('write').get('new_only')
        self.verbose = self.config.get('verbose')

        start = self.config['start']

        # dict of game files 
        self.files = {}

        self.step_num = 0
        self.log_path = self.dir + 'LOG.csv'
        if os.path.isfile(self.log_path):
            self.log_file = open(self.log_path, 'a')
        else:
            log_header = ['index', 'time', 'time_diff', 'num_changes']
            self.log_file = open(self.log_path, 'a')
            openers.write_list(self.log_file, log_header)
        self.files['LOG'] = self.log_file

        self.prev_time = time.time()

        if self.write_new:
            self.prevs = bov.lines(
                self.sports, verbose=self.verbose, output='dict')
            self.current = None

        if start:
            self.run()


    def step(self):
        '''

        '''
        self.new_time = time.time()
        self.current = bov.lines(self.sports, verbose=self.verbose, output='dict')

        if self.write_new:
            to_write = compare_and_filter(self.prevs, self.current)
            self.prevs = self.current
            time.sleep(self.wait)
            changes = len(to_write)
        else:
            to_write = self.current
            changes = "NaN"
        
        time_delta = self.new_time - self.prev_time
        self.prev_time = self.new_time

        self.files = write_data(self.files, to_write, verbose=self.verbose)
        self.step_num += 1
        
        log_data = [self.step_num, self.new_time, time_delta, changes]
        openers.write_list(self.log_file, log_data)
        

    def run(self):
        '''
        
        '''
        try:
            while True:
                self.step()
        except KeyboardInterrupt:
            print('interrupted')
            for fn, f in self.files.items():
                f.close()
                print(f'closed {fn}')


def compare_and_filter(prevs, news):
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


def write_data(file_dict, data_dict, verbose=True):
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
        if verbose:
            print(f'writing {v} to game [{k}]')

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
    line = Lines('./config/new_lines.json')
