import os
import time
import json

import sips.h.openers as io
from sips.lines.bov import bov
from sips.lines.bov.utils import bov_utils as utils
from sips.h import openers

from requests_futures.sessions import FuturesSession
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


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
        self.wait = self.config.get('wait')
        self.verbose = self.config.get('verbose')
        start = self.config['start']

        file_conf = self.config.get('file')
        self.write_new = file_conf.get('new_only')
        # self.file_per_game = file_conf.get('file_per_game')
        self.flush_rate = file_conf.get('flush_rate')
        self.keep_open = file_conf.get('keep_open')

        self.session = FuturesSession(
            executor=ProcessPoolExecutor(max_workers=10))

        # if keep_open, dict of files, else dict of file names
        self.files = {}
        # todo multiple ways of writing/grouping output

        self.step_num = 0
        self.log_path = self.dir + 'LOG.csv'
        if os.path.isfile(self.log_path):
            self.log_file = open(self.log_path, 'a')
        else:
            log_header = ['index', 'time', 'time_diff',
                          'num_events', 'num_changes']
            self.log_file = open(self.log_path, 'a')
            openers.write_list(self.log_file, log_header)
        self.files['LOG'] = self.log_file
        self.log_data = None
        self.prev_time = time.time()

        if self.write_new:
            self.prevs = bov.lines(
                self.sports, output='dict', verbose=self.verbose, session=self.session)
            self.current = None

        if start:
            self.run()

    def step(self):
        '''

        '''
        self.new_time = time.time()
        self.current = bov.lines(
            self.sports, verbose=self. verbose, output='dict', session=self.session)

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
        if self.keep_open:
            
        else:
            self.files = open_and_write(self.files, to_write, verbose=self.verbose)

        self.step_num += 1

        self.log_data = [self.step_num, self.new_time,
                         time_delta, len(self.current), changes]
        openers.write_list(self.log_file, self.log_data)
        self.flush_log_file()

    def run(self):
        '''

        '''
        try:
            while True:
                self.step()
        except KeyboardInterrupt:
            print('interrupted')

    def flush_log_file(self):
        if self.step_num % self.flush_rate == 1:
            print(f'{self.log_data}')
            self.log_file.flush()
            if self.keep_open:
                for game_file in self.files.values():
                    game_file.flush()


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


def init_file(fn, keep_open=False):
    f = open(fn, 'a')
    io.write_list(f, utils.header())
    if not keep_open:
        f.close()
    else:
        return f


def write_opened(file_dict, data_dict, verbose=True):
    '''
    read in dictionary with open files as values
    and write data to files
    '''
    for game_id, vals in data_dict.items():
        f = file_dict.get(game_id)

        if not f:
            fn = '../data/lines/' + str(game_id) + '.csv'
            f = init_file(fn, keep_open=True)
            file_dict[game_id] = f
        
        io.write_list(f, vals)
        if verbose:
            print(f'writing {vals} to game [{game_id}]')
    
    return file_dict


def open_and_write(file_dict, data_dict, verbose=True):
    '''
    read in dictionary of file names and compare to new data
    '''
    for game_id, vals in data_dict.items():
        f = file_dict.get(game_id)
        fn = '../data/lines/' + str(game_id) + '.csv'
        if not f:
            file_dict[game_id] = fn
            if not os.path.isfile(fn):
                init_file(fn)

        f = open(fn, 'a')

        io.write_list(f, vals)
        if verbose:
            print(f'writing {vals} to game [{game_id}]')

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
    line = Lines(
        '/home/sippycups/absa/sips/sips/lines/bov/config/new_lines.json')
