'''
this is the main runner for bovada

argparse args will overwrite the config values
'''

import os
import argparse
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import sips
import sips.h.fileio as io
from sips.ml import lstm
from sips.h.cloud import profiler
from sips.lines import collate
from sips.macros import macros as m
from sips.macros import bov as bm
from sips.lines.bov import bov
from sips.lines.bov.utils import bov_utils as utils

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from requests_futures.sessions import FuturesSession


LINES_DATA_PATH = m.PARENT_DIR + '/data/lines/'
CONFIG_PATH = m.PROJ_DIR + 'lines/config/lines.json'


parser = argparse.ArgumentParser(description='configure lines.py')
parser.add_argument('-d', '--dir', type=str,
                    help='folder name of run', default='run3')
group = parser.add_mutually_exclusive_group()
group.add_argument('-s', '--sports', type=list,
                   help='list of 3 letter sports', default=['basketball/nba'])  # , 'football/nfl', 'hockey/nhl'])
group.add_argument('-A', '--all', type=bool, help='run on all sports')
parser.add_argument('-m', '--all_mkts', type=bool, help='true grabs extra markets',
                    default=False)
parser.add_argument('-l', '--log', type=bool, help='add the gcloud profiler')
parser.add_argument('-n', '--new_only', type=bool, help='', default=True)
parser.add_argument('-w', '--wait', type=float,
                    help='how long to wait after each step', default=0.25)
parser.add_argument('-v', '--verbose', type=bool,
                    help='print more', default=False)
parser.add_argument('-c', '--grab_espn', type=bool,
                    help='collate with espn data', default=False)
parser.add_argument('-r', '--run', type=bool, help='run on init', default=True)
parser.add_argument('-u', '--unique', type=bool,
                    help='write each game to a unique file', default=True)
parser.add_argument('-k', '--keep_open', type=bool,
                    help='keep files open while running', default=False)
parser.add_argument('-f', '--flush_rate', type=int,
                    help='how often log is flushed, as well as the files if keep_open',
                    default=1)
parser.add_argument('--async_req', type=bool,
                    help='use async_req (broken)',
                    default=False)
args = parser.parse_args()

if args.log:
    profiler.main()


class Lines:
    '''
    sport (str):
        - used to construct the api urls

    wait: (positive number)
        - number of seconds to sleep between each request

    write_config: 
        - if true, data is only written if it is different than previous
        sport='football/nfl', wait=5, start=True, write_new=False, verbose=False
    '''

    def __init__(self, config_path=None):
        '''

        '''
        if config_path:
            with open(config_path) as config:
                self.config = json.load(config)

            self.conf_from_file()
        else:
            self.conf_from_args()
        self.init_fileio()
        print(f'sports: {self.sports}')

        self.prev_time = time.time()
        if self.write_new:
            if self.espn:
                self.prevs = collate.get_and_compare(sports=self.sports)
            else:
                self.prevs = bov.lines(self.sports, output='dict',
                                       all_mkts=self.all_mkts,
                                       verbose=self.verb)
            self.current = None

        self.model = lstm.LSTM()
        self.loss_fxn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.001, momentum=0.9)
        self.running_loss = 0.0
        self.correct = 0
        self.model_log_file = io.init_csv(
            'model_log.csv', header=['i', 'running_loss'], close=False)

        self.step_num = 0
        if self.start:
            self.run()

    def conf_from_args(self):
        '''

        '''
        if args.all:
            self.sports = bm.SPORTS
        else:
            self.sports = args.sports

        print(self.sports)
        self.wait = args.wait
        self.verb = args.verbose
        self.req_async = args.async_req
        self.start = args.run
        self.espn = args.grab_espn
        self.all_mkts = args.all_mkts
        self.write_new = args.new_only
        self.flush_rate = args.flush_rate
        self.keep_open = args.keep_open
        self.file_per_game = args.unique
        self.folder_name = args.dir
        self.dir = LINES_DATA_PATH + self.folder_name + '/'
        self.session = None

    def conf_from_file(self):
        '''

        '''
        file_conf = self.config.get('file')
        self.sports = self.config.get('sports')
        if self.sports == 'all':
            self.sports = bm.SPORTS

        self.wait = self.config.get('wait')
        self.verb = self.config.get('verbose')
        self.req_async = self.config.get('async_req')
        self.start = self.config.get('run')
        self.espn = self.config.get('grab_espn')
        self.all_mkts = self.config.get('all_mkts')
        self.write_new = file_conf.get('new_only')
        self.flush_rate = file_conf.get('flush_rate')
        self.keep_open = file_conf.get('keep_open')
        self.file_per_game = file_conf.get('file_per_game')
        self.folder_name = file_conf.get('folder_name')
        self.dir = LINES_DATA_PATH + self.folder_name + '/'

        if self.req_async:
            self.session = FuturesSession(
                executor=ProcessPoolExecutor())
        else:
            self.session = None

    def init_fileio(self):
        '''

        '''
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        self.files = {}
        self.log_path = self.dir + 'log.csv'
        if os.path.isfile(self.log_path):
            self.log_file = open(self.log_path, 'a')
        else:
            log_header = ['index', 'time', 'time_diff', 'num_events',
                          'num_changes']
            self.log_file = open(self.log_path, 'a')
            io.write_list(self.log_file, log_header)

        self.files['log'] = self.log_file
        self.log_data = None

    def step(self):
        '''

        '''
        self.new_time = time.time()
        if self.espn:
            self.current = collate.get_and_compare(sports=self.sports)
        else:
            self.current = bov.lines(self.sports, output='dict',
                                     all_mkts=self.all_mkts, verbose=self.verb)

        if self.write_new:
            to_write = compare_and_filter(self.prevs, self.current)
            self.run_model()
            self.prevs = self.current
            time.sleep(self.wait)
            num_changes = len(to_write)
        else:
            to_write = self.current
            num_changes = "NaN"

        time_delta = self.new_time - self.prev_time
        self.prev_time = self.new_time

        if self.keep_open:
            self.files = write_opened(self.dir,
                                      self.files, to_write, verbose=self.verb)
        else:
            self.files = open_and_write(self.dir,
                                        self.files, to_write, verbose=self.verb)

        self.step_num += 1

        self.log_data = [self.step_num, self.new_time,
                         time_delta, len(self.current), num_changes]
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
        '''

        '''
        if self.step_num % self.flush_rate == 1:
            print(f'{self.log_data}')
            self.log_file.flush()
            torch.save(self.model, f'live_{self.step_num}.pt')
            if self.keep_open:
                for game_file in self.files.values():
                    game_file.flush()

    def run_model(self):
        '''
        for each event, predict what the transition type
        todo: async exec
        '''
        for k, v in self.current.items():

            nls = []
            for line in [self.prevs, self.current]:
                a, h = line[k][16:18]
                nl = []
                for t in [a, h]:
                    try:
                        x = float(t)
                    except:
                        x = -1
                    nl.append(x)
                nls.append(nl)

            prev_mls, cur_mls = nls
            true_transition = bov.classify_transition(prev_mls, cur_mls)
            print(f'true: {true_transition}')

            X = torch.tensor(bov.serialize_row(self.prevs[k]))
            self.optimizer.zero_grad()

            yhat = self.model(X).view(1, -1)
            y = torch.tensor(true_transition).view(1, -1).long()
            
            # print(f'y: {y}, pred: {yhat}')
            # print(f'y.dtype: {y.dtype}, yhat.dtype: {yhat.dtype}')
            # print(f'y.shape: {y.shape}, yhat.shape: {yhat.shape}')

            loss = self.loss_fxn(yhat, torch.max(y, 1)[1])
            self.optimizer.step()

            class_preds = torch.argmax(yhat)

            # print statistics
            self.running_loss += loss.item()
            self.correct += (class_preds == y).sum().item()
            if self.step_num % self.flush_rate == 0:
                print(f'{self.step_num}: correct: {self.correct} \
                    loss: {self.running_loss / self.flush_rate}')

                io.write_list(self.model_log_file, [
                              self.step_num, self.running_loss, self.correct])
                self.running_loss = 0.0
                self.correct = 0.0


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


def write_opened(dir, file_dict, data_dict, verbose=True):
    '''
    read in dictionary with open files as values
    and write data to files
    '''
    for game_id, vals in data_dict.items():
        f = file_dict.get(game_id)

        if not f:
            fn = dir + str(game_id) + '.csv'
            f = io.init_csv(fn, header=bm.LINE_COLUMNS, close=False)
            file_dict[game_id] = f

        io.write_list(f, vals)
        if verbose:
            print(f'writing {vals} to game [{game_id}]')

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
                io.init_csv(fn, header=bm.LINE_COLUMNS)

        f = open(fn, 'a')

        io.write_list(f, vals)
        if verbose:
            print(f'writing {vals} to game [{game_id}]')

        f.close()
    return file_dict


def main():
    # using config
    # sips_path = sips.__path__[0] + '/'
    # bov_lines = Lines(sips_path + 'lines/config/lines.json')

    # using argparse
    bov_lines = Lines()
    return bov_lines


if __name__ == '__main__':
    main()
