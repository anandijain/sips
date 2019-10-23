import time
import json 

from sips.lines.bov import bov

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

        with open(config_path) as config:
            self.config = json.load(config)
        
        self.sport = self.config.get('sport')
        self.wait = self.config.get('wait')
        self.write_new = self.config.get('write').get('new_only')
        self.verbose = self.config.get('verbose')

        start = self.config['start']

        # dict of game files 
        self.files = {}

        if start:
            self.run()

    def run(self):
        try:
            prevs = bov.lines(self.sport)
            while True:
                news = bov.lines(self.sport)
                to_write = prepare_write(prevs, news)
                write_data(self.files, to_write)
                prevs = news
                time.sleep(self.wait)
        except KeyboardInterrupt:
            print('interrupted')


def prepare_write(prevs, news):
    '''
    input: both are dictionaries of (game_id, event) 

    returns: dictionary of (game_id, row_data) of new data
    '''

    to_write = {}

    for k, v in news:
        if not prevs.get(k) or prevs.get(k) != v:
            to_write[k] = v
        else:
            continue
    return to_write


def write_data(file_dict, data_dict):
    '''

    '''
    for k, v in data_dict.items():
        f = file_dict.get(k)
        fn = '../data/' + str(k) + '.csv'
        if not f:
            file_dict[k] = fn

        f = open(fn, 'a')
        f.write(v)
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
