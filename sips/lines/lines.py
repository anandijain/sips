import os
import time
import json

import sips
import sips.h.openers as io
from sips.lines.bov import bov
from sips.lines.bov.utils import bov_utils as utils
from sips.lines import collate

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from requests_futures.sessions import FuturesSession

CONFIG_PATH = sips.__path__[0] + '/' + 'lines/config/lines.json'


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

    def __init__(self, config_path=CONFIG_PATH):
        '''

        '''

        with open(config_path) as config:
            self.config = json.load(config)

        self.conf()
        print(f'sports: {self.sports}')

        if self.req_async:
            self.session = FuturesSession(
                executor=ProcessPoolExecutor())
        else:
            self.session = None

        self.files = {}
        self.log_path = self.dir + 'LOG.csv'
        if os.path.isfile(self.log_path):
            self.log_file = open(self.log_path, 'a')
        else:
            log_header = ['index', 'time', 'time_diff', 'num_events',
                          'num_changes']
            self.log_file = open(self.log_path, 'a')
            io.write_list(self.log_file, log_header)

        self.files['LOG'] = self.log_file
        self.log_data = None

        self.prev_time = time.time()
        if self.write_new:
            if self.espn:
                self.prevs = collate.get_and_compare(sports=self.sports)
            else:
                self.prevs = bov.lines(self.sports, output='dict',
                                       verbose=self.verbose, espn=self.espn)
            self.current = None

        self.step_num = 0
        if self.start:
            self.run()

    def conf(self):
        '''

        '''
        file_conf = self.config.get('file')
        self.sports = self.config.get('sports')
        self.wait = self.config.get('wait')
        self.verbose = self.config.get('verbose')
        self.req_async = self.config.get('async_req')
        self.start = self.config.get('start')
        self.espn = self.config.get('grab_espn')
        self.write_new = file_conf.get('new_only')
        self.flush_rate = file_conf.get('flush_rate')
        self.keep_open = file_conf.get('keep_open')
        # self.file_per_game = file_conf.get('file_per_game')  todo
        self.folder_name = file_conf.get('folder_name')
        sips_path = sips.__path__[0] + '/'
        self.dir = sips_path + 'lines/data/lines/' + self.folder_name + '/'

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def step(self):
        '''

        '''
        self.new_time = time.time()
        if self.espn:
            self.current = collate.get_and_compare(sports=self.sports)
        else:
            self.current = bov.lines(self.sports, verbose=self.verbose,
                                     output='dict', espn=self.espn)

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
            self.files = write_opened(
                self.files, to_write, verbose=self.verbose)
        else:
            self.files = open_and_write(self.dir,
                self.files, to_write, verbose=self.verbose)

        self.step_num += 1

        self.log_data = [self.step_num, self.new_time,
                         time_delta, len(self.current), changes]
        io.write_list(self.log_file, self.log_data)
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


def write_opened(file_dict, data_dict, verbose=True):
    '''
    read in dictionary with open files as values
    and write data to files
    '''
    for game_id, vals in data_dict.items():
        f = file_dict.get(game_id)

        if not f:
            fn = sips.__path__[0] + '/' + str(game_id) + '.csv'
            f = io.init_csv(fn, header=utils.header(), close=False)
            file_dict[game_id] = f

        io.write_list(f, vals)
        if verbose:
            print(f'writing {vals} to game [{game_id}]')

    return file_dict


# def async_write_opened(file_dict, data_dict, verbose=True):
#     '''
#     read in dictionary with open files as values
#     and write data to files
#     '''

#     args = ((check_if_exists(file_dict, game_id), game_id, val)
#             for game_id, val in data_dict.items())

#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         results = [executor.map(write_for_map, args)]
#         as_completed = concurrent.futures.as_completed(results)
#         file_dict = as_completed[-1]

#     return file_dict


def write_for_map(file_dict, game_id, vals):
    file_dict = check_if_exists(file_dict, game_id)
    f = file_dict[game_id]
    io.write_list(f, vals)
    return file_dict


def check_if_exists(file_dict, key):
    '''
    given a dictionary and a key
    if key not in dict, init file, add to dict, return updated dict
    '''
    f = file_dict.get(key)

    if not f:
        fn = '../data/lines/' + str(key) + '.csv'
        f = io.init_csv(fn, header=utils.header(), close=False)
        file_dict[key] = f
    return file_dict


def open_and_write(dir, file_dict, data_dict, verbose=True):
    '''
    read in dictionary of file names and compare to new data
    '''
    for game_id, vals in data_dict.items():
        f = file_dict.get(game_id)
        fn = dir + str(game_id) + '.csv'
        if not f:
            file_dict[game_id] = fn
            if not os.path.isfile(fn):
                io.init_csv(fn, header=utils.header())

        f = open(fn, 'a')

        io.write_list(f, vals)
        if verbose:
            print(f'writing {vals} to game [{game_id}]')

        f.close()
    return file_dict


def main():
    sips_path = sips.__path__[0] + '/'
    bov_lines = Lines(sips_path + 'lines/config/lines.json')
    return bov_lines


if __name__ == '__main__':
    main()
