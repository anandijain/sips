import os
import os.path

import time
import requests

import matplotlib.pyplot as plt

import numpy as np

import sips.gym_sip.h as h


save_path = 'data'
root_url = 'https://www.bovada.lv'
scores_url = "https://services.bovada.lv/services/sports/results/api/v1/scores/"
headers = {'User-Agent': 'Mozilla/5.0'}

# TODO verbosity fix


class Sippy:
    def __init__(self, fn=None, header=0, league=0, verbosity=True):
        print("~~~~sippywoke~~~~")
        self.games = []
        self.events = []

        self.links = []
        self.all_urls = h.macros.build_urls()

        self.verbosity = verbosity
        self.league = league
        self.set_league(self.league)
        self.json_events()

        self.counter = 0
        self.num_updates = 0
        self.twenty_steps_ago = 0

        if fn is not None:
            self.file = open_file(fn)
            self.file.flush()
        else:
            self.file = None


        access_time = time.time()
        self.init_games(access_time)

    def step(self):
        access_time = time.time()

        self.json_events()
        self.cur_games(access_time)

        print('step: {}'.format(self.counter))
        print('time: {}\n'.format(time.asctime()))

        self.counter += 1

        if self.counter % 20 == 1:
            elapsed = round(abs(access_time - self.twenty_steps_ago))

            if elapsed == 0:
                efficiency = 0
            else:
                efficiency = self.num_updates/elapsed

            print("num games: " + str(len(self.games)))
            print('num events: ' + str(len(self.events)))
            print('new lines in past 20 steps: {} / {} seconds'.format(self.num_updates, elapsed))
            print('rough efficiency (newlines/elapsed): {}\n'.format(efficiency))

            self.twenty_steps_ago = access_time
            self.num_updates = 0
            self.update_games_list()
            if self.file is not None:
                self.file.flush()

        for game in self.games:
            if game.score.ended == 0:  # if game is not over
                if game.lines.updated == 1 or game.score.new == 1:  # if lines updated or score updated

                    if self.file is not None:
                        game.write_game(self.file)  # write to csv

                    if self.verbosity:
                        print(game)

                    self.num_updates += 1
                    game.lines.updated = 0  # reset lines to not be updated
                    game.score.new == 0
                if game.score.a_win == 1 or game.score.h_win == 1:
                    game.score.ended = 1

    def cur_games(self, access_time):
        for event in self.events:
            exists = 0
            for game in self.games:
                if event['id'] == game.game_id:
                    if game.score.ended == 1:
                        continue
                    game.lines.update(event)
                    game.score.update()
                    exists = 1
                    break
            if exists == 0:
                self.new_game(event, access_time)

    def json_events(self):
        pages = []
        events = []
        for link in self.links:
            pages.append(req(link))
            if self.verbosity is True:
                print('.', end='')

        for page in pages:
            try:
                for section in page:
                    league = section['path'][0]['description']
                    tmp = section.get('events')
                    for event in tmp:
                        link = event.get('link')
                        if link is not None:
                            event.update({'league': league})
                            events.append(event)
            except TypeError:
                pass
        self.events = events

    def update_games_list(self):
        for game in self.games:
            in_json = 0
            for event in self.events:
                if game.game_id == event['id']:
                    in_json = 1
                    break
            if in_json == 0:
                self.games.remove(game)

    def info(self, verbose):  # 1 for verbose, else for abridged
        print(str(len(self.games)))
        for game in self.games:
            if verbose == 1:
                try:
                    print(game)
                except TypeError:
                    continue
            else:
                game.quick()

    def new_game(self, event, access_time):
        x = Game(event, access_time, self.league)
        # x.quick()
        self.games.insert(0, x)

    def init_games(self, access_time):
        for event in self.events:
            self.new_game(event, access_time)

    def id_given_teams(self, a_team, h_team):  # input is two strings
        for game in self.games:
            if game.a_team == a_team and game.h_team == h_team:
                return game.game_id
            else:
                return None

    def run(self):
        while True:
            self.step()

    def set_league(self, league):
        str_league = str(league).lower()
        if league == 0 or str_league == 'all':
            self.links = self.all_urls
        elif league == 1 or str_league == 'nba':
            self.links = self.all_urls[0:2]
        elif league == 2 or str_league == 'college basketball':
            self.links = self.all_urls[2:4]
        elif league == 3 or str_league == 'mlb':
            self.links = self.all_urls[4:6]
        elif league == 4 or str_league == 'esports':
            self.links = self.all_urls[6:8]
        elif league == 5 or str_league == 'football':
            self.links = self.all_urls[8:10]
        elif league == 6 or str_league == 'tennis':
            self.links = self.all_urls[10:12]
        elif league == 7 or str_league == 'volleyball':
            self.links = self.all_urls[12:14]
        elif league == 8 or str_league == 'hockey':
            self.links = self.all_urls[14:16]
        else:
            self.links = self.all_urls[4:6]

    def write_header(self):
        num_headers = len(h.macros.bovada_headers)
        for i, column_header in enumerate(h.macros.bovada_headers):
            if i < num_headers:
                self.file.write(column_header + ',')
            else:
                self.file.write(column_header)
        self.file.write('\n')

    def __repr__(self):
        for game in self.games:
            try:
                print(game)
            except TypeError:
                return '.'


class Game:
    def __init__(self, event, access_time, gtype):
        self.init_time = access_time
        self.sport = event['sport']
        self.gtype = gtype
        self.league = event.get('league')
        self.league_fix()
        self.game_id = event['id']
        self.desc = event['description']
        if self.gtype == 4 or self.gtype == 7:
            sep = 'vs'
        else:
            sep = '@'
        try:
            self.a_team = self.desc.split(sep)[0]
            self.a_team = self.a_team[:-1]
            self.h_team = self.desc.split(sep)[1]
            self.h_team = self.h_team[1:]
            self.teams = [self.a_team, self.h_team]
        except Exception:
            self.a_team = self.desc
            self.h_team = self.desc
        self.start_time = event['startTime'] / 1000.
        self.score = Score(self.game_id)
        self.lines = Lines(event)
        self.link = event['link']
        self.delta = None

    def write_game(self, file):
        self.time_diff()
        file.write(self.sport + ',')
        file.write(self.league + ',')
        file.write(self.game_id + ',')
        file.write(self.a_team + ',')
        file.write(self.h_team + ',')
        file.write(str(time.time()) + ',')
        self.score.csv(file)
        file.write(str(self.delta) + ',')
        self.lines.csv(file)
        file.write(str(self.start_time) + "\n")

    def return_row(self):
        self.time_diff()
        row = [self.sport, self.league, self.game_id, self.a_team, self.h_team, str(time.time())]
        row += self.score.jps
        row.append(self.delta)
        row += self.lines.jps
        row.append(self.start_time)
        return row

    def quick(self):
        print(str(self.lines.last_mod_lines))
        print(self.a_team, end=': ')
        print(str(self.score.a_pts) + ' ' + str(self.lines.a_ml))
        print(self.h_team, end=': ')
        print(str(self.score.h_pts) + ' ' + str(self.lines.h_ml))

    def score(self):
        print(self.a_team + " " + str(self.score.a_pts))
        print(self.h_team + " " + str(self.score.h_pts))

    def odds(self):
        print(self.a_team + " " + str(self.lines.odds()))
        print(self.h_team + " " + str(self.lines.odds()))

    def time_diff(self):
        if len(self.lines.last_mod_lines) > 0:
            self.delta = (self.lines.last_mod_lines[-1] - self.start_time)
        else:
            self.delta = '0'

    def league_fix(self):
        if self.league is not None:
            self.league = self.league.replace(',', '')

    def __repr__(self):
        try:
            print('\n' + self.desc + '\n')
            print(self.sport, end='|')
            print(self.game_id, end='|')
            print(self.a_team, end='|')
            print(self.h_team, end='|')
            print('\nScores info: ')
            print(self.score)
            print('Game line info: ')
            print('time delta: {} |'.format(self.delta))
            print(self.lines)
            print(str(self.start_time) + "\n")
        except TypeError:
            return '.'

class Lines:
    def __init__(self, json):
        self.updated = 0
        self.json = json
        self.jps = []
        self.mkts = []
        [self.query_times, self.last_mod_lines, self.num_markets, self.a_ml, self.h_ml, self.a_deci_ml,
         self.h_deci_ml, self.a_odds_ps, self.h_odds_ps, self.a_deci_ps, self.h_deci_ps, self.a_hcap_ps,
         self.h_hcap_ps, self.a_odds_tot, self.h_odds_tot, self.a_deci_tot, self.h_deci_tot, self.a_hcap_tot,
         self.h_hcap_tot, self.a_ou, self.h_ou] = ([] for i in range(21))

        self.params = [self.last_mod_lines, self.num_markets, self.a_ml, self.h_ml, self.a_deci_ml,
                        self.h_deci_ml, self.a_odds_ps, self.h_odds_ps, self.a_deci_ps, self.h_deci_ps,
                        self.a_hcap_ps, self.h_hcap_ps, self.a_odds_tot, self.h_odds_tot, self.a_deci_tot,
                        self.h_deci_tot, self.a_hcap_tot, self.h_hcap_tot, self.a_ou, self.h_ou]

    def update(self, json):
        self.updated = 0
        self.json = json
        self.jparams()

        if len(self.params[0]) > 0:
            if self.jps[0] == self.params[0][-1]:
                self.updated = 0
                return

        i = 0
        for param in self.params:
            if self.jps[i] is None:
                self.jps[i] = 0
            if len(param) > 0:
                if param[-1] == self.jps[i]:
                    i += 1
                    continue
            self.params[i].append(self.jps[i])
            self.updated = 1
            i += 1

    def jparams(self):
        j_markets = self.json['displayGroups'][0]['markets']
        data = {"american": 0, "decimal": 0, "handicap": 0}

        a_ou = None
        h_ou = None

        self.mkts = []

        ps = Market(data, data)
        ml = Market(data, data)
        tot = Market(data, data)

        self.mkts += [ps, ml, tot]

        for market in j_markets:
            outcomes = market['outcomes']
            desc = market.get('description')

            try:
                away_price = outcomes[0].get('price')
            except IndexError:
                away_price = data
            try:
                home_price = outcomes[1].get('price')
            except IndexError:
                home_price = data

            if desc is None:
                continue
            elif desc == 'Point Spread' or desc == 'Runline' or desc == 'Puck Line':
                ps.update(away_price, home_price)
            elif desc == 'Moneyline':
                ml.update(away_price, home_price)
            elif desc == 'Total':
                try:
                    a_ou = outcomes[0].get('type')
                    h_ou = outcomes[1].get('type')
                except IndexError:
                    a_ou = '0'
                    h_ou = '0'
                tot.update(away_price, home_price)

        self.even_handler()

        last_mod = self.json['lastModified'] / 1000.  # conversion from ns

        # shape jps to always be 18 elements long for now via adding extra elements to a list that is to short

        self.jps = [last_mod, self.json['numMarkets'], self.mkts[1].a['american'], self.mkts[1].h['american'],
                    self.mkts[1].a['decimal'], self.mkts[1].h['decimal'], self.mkts[0].a['american'], self.mkts[0].h['american'],
                    self.mkts[0].a['decimal'], self.mkts[0].h['decimal'], self.mkts[0].a['handicap'], self.mkts[0].h['handicap'],
                    self.mkts[2].a['american'], self.mkts[2].h['american'], self.mkts[2].a['decimal'], self.mkts[2].h['decimal'],
                    self.mkts[2].a['handicap'], self.mkts[2].h['handicap'], a_ou, h_ou]

    def even_handler(self):
        for mkt in self.mkts:
            if mkt.a['american'] == 'EVEN' and mkt.h['american'] == 'EVEN':
                mkt.a['american'] = 100
                mkt.h['american'] = 100
            elif mkt.a['american'] == 'EVEN':
                if int(mkt.h['american']) > 0:
                    mkt.a['american'] = -100
                elif int(mkt.h['american']) < 0:
                    mkt.a['american'] = 100
                else:
                    mkt.a['american'] = 0
            elif mkt.h['american'] == 'EVEN':
                if int(mkt.a['american']) > 0:
                    mkt.h['american'] = -100
                elif int(mkt.a['american']) < 0:
                    mkt.h['american'] = 100
                else:
                    mkt.h['american'] = 0

    def csv(self, file):
        for param in self.params:
            if len(param) > 0:
                file.write(str(param[-1]))
                file.write(",")
            else:
                file.write(str(0))
                file.write(',')

    def odds(self):
        for elt in [self.last_mod_lines, self.a_ml, self.h_ml]:
            print(str(elt), end='|')

    def __repr__(self):
        for param in self.params:
            try:
                print(str(param[-1]), end='|')
            except IndexError:
                print('None', end='|')
        print('\n')


class Score:
    def __init__(self, game_id):
        self.new = 1
        self.game_id = game_id
        self.num_quarters = 0
        self.dir_isdown = 0
        self.jps = []
        self.data = None
        self.clock = None
        self.json()
        self.jparams()
        self.ended = 0

        [self.lms_date, self.lms_time, self.quarter, self.secs, self.a_pts, self.h_pts,
            self.status, self.a_win, self.h_win] = ([] for i in range(9))

        self.params = [self.lms_date, self.lms_time, self.quarter, self.secs, self.a_pts,
                       self.h_pts, self.status, self.a_win, self.h_win]

    def update(self):
        self.new = 0
        self.json()
        if self.data is None:
            return
        self.clock = self.data.get('clock')
        if self.clock is None:
            return
        self.jparams()
        self.metadata()
        self.win_check()

    def metadata(self):
        if self.same() == 1:
            return
        i = 0
        for jp in self.jps:
            if len(self.params[i]) > 0:
                if self.params[i][-1] == self.jps[i]:
                    i += 1
                    continue
            self.params[i].append(jp)
            self.new = 1
            i += 1

    def jparams(self):
        if self.data is None:
            return
        self.clock = self.data.get('clock')
        if self.clock is None:
            return
        status = 0
        if self.data['gameStatus'] == "IN_PROGRESS":
            status = 1

        score = self.data.get('latestScore')
        dt = self.date_split()

        self.jps = [dt[0], dt[1], self.clock.get('periodNumber'),
                    self.clock.get('relativeGameTimeInSecs'), score.get('visitor'), score.get('home'), status]

        self.num_quarters = self.clock.get('numberOfPeriods')
        self.dir_isdown = self.clock.get('direction')

    def win_check(self):
        # a = self.quarter[-1] == 0 or self.quarter[-1] == -1
        # b = self.quarter[-1] == self.num_quarters
        # c = self.secs[-1] == 0 or self.secs[-1] == -1
        # d = self.status[-1] == 0
        # win = b and c and d
        # win2 = a and c and d
        # print('ended {}'.format(self.ended))
        # print('last q {}'.format(self.quarter[-1]))
        # print('status {}\n'.format(self.status[-1]))

        if self.ended == 0:
            if self.quarter[-1] == -1 and self.status[-1] == 0:
                if self.a_pts[-1] > self.h_pts[-1]:
                    self.a_win.append(1)
                    self.h_win.append(0)
                    print('a_team win')

                elif self.h_pts[-1] > self.a_pts[-1]:
                    self.a_win.append(0)
                    self.h_win.append(1)
                    print('h_team win')

    def date_split(self):
        dt = self.data['lastUpdated'].split('T')
        return dt

    def same(self):
        dt = self.date_split()
        if len(self.lms_date) > 0 and len(self.lms_time) > 0:
            if self.lms_date[-1] == dt[0] and self.lms_time[-1] == dt[1]:
                self.new = 0
                return 1

    def csv(self, file):
        for value in self.values():
                file.write(value + ',')

    def values(self):
        data = []
        for param in self.params:
            if len(param) > 0:
                if param is None:
                    param = ''
                data.append(str(param[-1]))
            else:
                data.append('0')
        return data

    def json(self):
        self.data = req(scores_url + self.game_id)

    def __repr__(self):
        for i, param in enumerate(self.params):
            if len(param) > 0:
                print('{}: {}'.format(h.macros.bov_score_headers[i], param[-1]), end='|')
        print('\n')


class Market:
    def __init__(self, away, home):
        self.frame = {"american": 0, "decimal": 0, "handicap": 0}
        self.a = away
        self.h = home

    def update(self, away, home):
        teams = [self.a, self.h]
        for i, team in enumerate([away, home]):
            for key in team:
                val = team[key]
                teams[i][key] = val

        self.a = away
        self.h = home

    def __repr__(self):
        print('away: {}'.format(self.a))
        print('home: {}'.format(self.h))

def req(url):
    try:
        r = requests.get(url, headers=headers, timeout=10)
    except ConnectionResetError:
        print('connection reset error')
        time.sleep(2)
        return
    except requests.exceptions.Timeout:
        print('requests.exceptions timeout error')
        time.sleep(2)
        return
    except requests.exceptions.ConnectionError:
        print('connectionerror')
        time.sleep(2)
        return
    try:
        return r.json()
    except ValueError:
        time.sleep(2)
        return


def open_file(file_name):
    complete_name = os.path.join(save_path, file_name + ".csv")
    exists = os.path.isfile(complete_name)

    if exists:  # don't write header
        file = open(complete_name, "a", encoding="utf-8")  # line below probably better called in caller or add another arg
    else:  # write header
        file = open(complete_name, "a", encoding="utf-8")  # line below probably better called in caller or add another arg
        write_header(file)

    return file


def write_json(file_name, json):
    file = open_file(file_name)
    file.write(json)
    file.write('\n')
    file.close()


def write_header(file):
    num_headers = len(h.macros.bovada_headers)
    for i, column_header in enumerate(h.macros.bovada_headers):
        if i < num_headers:
            file.write(column_header + ',')
        else:
            file.write(column_header)
    file.write('\n')
