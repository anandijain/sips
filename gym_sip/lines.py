import time
import os.path
import requests
import argparse
import helpers as h
import matplotlib.pyplot as plt 
import numpy as np


save_path = 'data'
root_url = 'https://www.bovada.lv'
scores_url = "https://services.bovada.lv/services/sports/results/api/v1/scores/"
headers = {'User-Agent': 'Mozilla/5.0'}

# TODO convert last_mod_score to epoch
# TODO add restart/timeout
# TODO 'EVEN' fix
# TODO fix scores class so that it stores a list of scores.
# TODO write the league, so as to differentiate between college and NBA
# TODO add short circuit to the score updater, if last_mod_score == cur last mod score, then return.
# TODO upon removing games that are no longer in json, this is a good point to calculate the actual profit of RL bot
# TODO add feature to csv that is a binary saying if information is 0/missing. it will help correct
#  for not knowing when lines close
# TODO add Over Under field and use it for one hot encoding


class Sippy:
    def __init__(self, fn='./data/write.csv', header=0, league=1):
        print("~~~~sippywoke~~~~")
        self.games = []
        self.links = []
        self.events = []
        self.x_axis = []
        self.y_axis = []
        self.league = league
        self.set_league(self.league)
        self.json_events()
        self.counter = 0
        
        if fn is not None:
            self.file = open_file(fn)
            if header == 1:
                self.write_header()

            self.file.flush()
        else:
            self.file = None

        access_time = time.time()
        self.init_games(access_time)

    def step(self):
        access_time = time.time()
        self.json_events()
        self.cur_games(access_time)
        time.sleep(1)

        print(str(self.counter) + ": time: " + str(time.localtime()) + '\n')
        
        self.counter += 1

        if self.counter % 20 == 1:
            print("num games: " + str(len(self.games)))
            print('num events: ' + str(len(self.events)))
            self.update_games_list()
            if self.file is not None:
                self.file.flush()

        for game in self.games:
            if game.score.ended == 0:  # if game is not over
                if game.lines.updated == 1 or game.score.new == 1:  # if lines updated or score updated
                    
                    if self.file is not None:
                        game.write_game(self.file)  # write to csv

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
        for page in pages:
            try:
                for section in page:
                    league = section['path'][0]['description']
                    tmp = section.get('events')
                    for event in tmp:
                        event.update({'league': league})
                    events += tmp
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
                game.info()
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
        if league == 1 or str_league == 'nba':  # NBA
            self.links = ["https://www.bovada.lv/services/sports/event/v2/events/A/" 
                          "description/basketball/nba?marketFilterId=def&liveOnly=true&lang=en",
                          "https://www.bovada.lv/services/sports/event/v2/events/" 
                          "A/description/basketball/nba?marketFilterId=def&preMatchOnly=true&lang=en"]
        elif league == 2 or str_league == 'college':  # College Bask
            self.links = ['https://www.bovada.lv/services/sports/event/v2/events/A/'
                          'description/basketball/college-basketball?marketFilterId=def&liveOnly=true&lang=en',
                          'https://www.bovada.lv/services/sports/event/v2/events/A/'
                          'description/basketball/college-basketball?marketFilterId=def'
                          '&preMatchOnly=true&eventsLimit=50&lang=en']
        elif league == 3 or str_league == 'hockey':  # Hockey
            self.links = ['https://www.bovada.lv/services/sports/event/v2/events/A/'
                          'description/hockey?marketFilterId=def&liveOnly=true&lang=en',
                          'https://www.bovada.lv/services/sports/event/v2/events/A/'
                          'description/hockey?marketFilterId=def&preMatchOnly=true&eventsLimit=50&lang=en']
        elif league == 4 or str_league == 'tennis':  # Tennis
            self.links = ['https://www.bovada.lv/services/sports/event/v2/events/A/'
                          'description/tennis?marketFilterId=def&liveOnly=true&eventsLimit=8&lang=en',
                          'https://www.bovada.lv/services/sports/event/v2/events/A/'
                          'description/tennis?marketFilterId=def&preMatchOnly=true&eventsLimit=50&lang=en']
        elif league == 5 or str_league == 'esports':  # Esports
            self.links = ['https://www.bovada.lv/services/sports/event/v2/events/A/'
                          'description/esports?marketFilterId=def&liveOnly=true&eventsLimit=8&lang=en',
                          'https://www.bovada.lv/services/sports/event/v2/events/A/'
                          'description/esports?marketFilterId=def&preMatchOnly=true&eventsLimit=50&lang=en']
        elif league == 6 or str_league == 'football':  # Football
            self.links = ['https://www.bovada.lv/services/sports/event/v2/events/A/'
                          'description/football?marketFilterId=def&liveOnly=true&eventsLimit=8&lang=en',
                          'https://www.bovada.lv/services/sports/event/v2/events/A/'
                          'description/football?marketFilterId=def&preMatchOnly=true&eventsLimit=50&lang=en']
        elif league == 7 or str_league == 'volleyball':  # Volleyball
            self.links = ['https://www.bovada.lv/services/sports/event/v2/events/A/'
                          'description/volleyball?marketFilterId=def&liveOnly=true&eventsLimit=8&lang=en',
                          'https://www.bovada.lv/services/sports/event/v2/events/A/'
                          'description/volleyball?marketFilterId=def&preMatchOnly=true&eventsLimit=50&lang=en']
        else:               # All BASK
            self.links = ["https://www.bovada.lv/services/sports/event/v2/events/A/" 
                          "description/basketball?marketFilterId=def&liveOnly=true&eventsLimit=8&lang=en",
                          "https://www.bovada.lv/services/sports/event/v2/events/A/" 
                          "description/basketball?marketFilterId=def&preMatchOnly=true&eventsLimit=50&lang=en"]

    def write_header(self):
        self.file.write("sport,league,game_id,a_team,h_team,cur_time,")
        self.file.write("lms_date,lms_time,quarter,secs,a_pts,h_pts,status,a_win,h_win,last_mod_to_start,")
        self.file.write("last_mod_lines,num_markets,a_odds_ml,h_odds_ml,a_deci_ml,h_deci_ml,")
        self.file.write("a_odds_ps,h_odds_ps,a_deci_ps,h_deci_ps,a_hcap_ps,h_hcap_ps,a_odds_tot,")
        self.file.write("h_odds_tot,a_deci_tot,h_deci_tot,a_hcap_tot,h_hcap_tot,")
        self.file.write("game_start_time\n")

    def __repr__(self):
        for game in self.games:
            print(game)


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
        self.a_team = self.desc.split(sep)[0]
        self.a_team = self.a_team[:-1]
        self.h_team = self.desc.split(sep)[1]
        self.h_team = self.h_team[1:]
        self.teams = [self.a_team, self.h_team]
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
        print(row)
        return row

    def info(self):  # displays scores, lines
        print('\n' + self.desc + '\n')
        print(self.sport, end='|')
        print(self.game_id, end='|')
        print(self.a_team, end='|')
        print(self.h_team, end='|')
        print('\nScores info: ')
        print(self.score)
        print('Game line info: ')
        print(str(self.delta), end='|')
        print(self.lines)
        print(str(self.start_time) + "\n")

    def quick(self):
        print(str(self.lines.last_mod_lines))
        print(self.a_team, end=': ')
        print(str(self.score.a_pts) + ' ' + str(self.lines.a_odds_ml))
        print(self.h_team, end=': ')
        print(str(self.score.h_pts) + ' ' + str(self.lines.h_odds_ml))

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
        self.info()


class Lines:
    def __init__(self, json):
        self.updated = 0
        self.json = json
        self.jps = []
        self.mkts = []
        [self.query_times, self.last_mod_lines, self.num_markets, self.a_odds_ml, self.h_odds_ml, self.a_deci_ml,
         self.h_deci_ml, self.a_odds_ps, self.h_odds_ps, self.a_deci_ps, self.h_deci_ps, self.a_hcap_ps,
         self.h_hcap_ps, self.a_odds_tot, self.h_odds_tot, self.a_deci_tot, self.h_deci_tot, self.a_hcap_tot,
         self.h_hcap_tot] = ([] for i in range(19))

        self.params = [
                     self.last_mod_lines, self.num_markets, self.a_odds_ml, self.h_odds_ml, self.a_deci_ml,
                     self.h_deci_ml, self.a_odds_ps, self.h_odds_ps, self.a_deci_ps, self.h_deci_ps,
                     self.a_hcap_ps, self.h_hcap_ps, self.a_odds_tot, self.h_odds_tot, self.a_deci_tot,
                     self.h_deci_tot, self.a_hcap_tot, self.h_hcap_tot
                      ]

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
        data2 = {"american": 0, "decimal": 0, "handicap": 0}
        self.mkts = []
        ps = Market(data, data2)
        self.mkts.append(ps)
        ml = Market(data, data2)
        self.mkts.append(ml)
        tot = Market(data, data2)
        self.mkts.append(tot)

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
                home_price = data2

            if desc is None:
                continue
            elif desc == 'Point Spread':
                ps.update(away_price, home_price)
            elif desc == 'Moneyline':
                ml.update(away_price, home_price)
            elif desc == 'Total':
                tot.update(away_price, home_price)

        self.even_handler()
        last_mod = self.json['lastModified'] / 1000.
        self.jps = [last_mod, self.json['numMarkets'], self.mkts[1].a['american'], self.mkts[1].h['american'],
                    self.mkts[1].a['decimal'], self.mkts[1].h['decimal'], self.mkts[0].a['american'], self.mkts[0].h['american'],
                    self.mkts[0].a['decimal'], self.mkts[0].h['decimal'], self.mkts[0].a['handicap'], self.mkts[0].h['handicap'],
                    self.mkts[2].a['american'], self.mkts[2].h['american'], self.mkts[2].a['decimal'], self.mkts[2].h['decimal'],
                    self.mkts[2].a['handicap'], self.mkts[2].h['handicap']]

    def even_handler(self):
        for mkt in self.mkts:
            if mkt.a['american'] == 'EVEN':
                if int(mkt.h['american']) > 0:
                    mkt.a['american'] = -100
                elif int(mkt.h['american']) < 0:
                    mkt.a['american'] = 100
                else:
                    mkt.a['american'] = 0

            if mkt.h['american'] == 'EVEN':
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
        for elt in [self.last_mod_lines, self.a_odds_ml, self.h_odds_ml]:
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
        # self.a_win.append(0)
        # self.h_win.append(0)

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
        a = self.quarter[-1] == 0
        b = self.quarter[-1] == self.num_quarters
        c = self.secs[-1] == 0
        d = self.status[-1] == 0
        win = b and c
        win2 = a and c and d

        if self.ended == 0:
            if win or win2:
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
        for param in self.params:
            if len(param) > 0:
                if param is None:
                    param = ''
                file.write(str(param[-1]) + ',')
            else:
                file.write('0' + ',')

    def json(self):
        self.data = req(scores_url + self.game_id)

    def __repr__(self):
        for param in self.params:
            if param is None:
                param = 0
            print(str(param), end='|')
        print('\n')


class Market:
    def __init__(self, away, home):
        self.a = away
        self.h = home

    def update(self, away, home):
        self.a = away
        self.h = home


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
    file = open(complete_name, "a", encoding="utf-8")  # line below probably better called in caller or add another arg
    return file


def write_json(file_name, json):
    file = open_file(file_name)
    file.write(json)
    file.write('\n')
    file.close()
