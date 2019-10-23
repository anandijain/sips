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

        # dict of game files 
        self.files = {}

        if start:
            self.run()

    def run(self):
        try:
            while True:
                events = bov.get_events(self.sport, self.verbose)
        except KeyboardInterrupt:
            print('interrupted')

def update_check(prev, new):
    '''
    prev - list of old data
    new - list of new data
    returns - bool
    '''
    is_updated = True if prev == new else False
    return is_updated
