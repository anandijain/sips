import requests 
import h
import time


HEADERS = {'User-Agent': 'Mozilla/5.0'}

class BovadaState:
	def __init__(self):
		# # self.links = ["https://www.bovada.lv/services/sports/event/v2/events/A/" 
  # #                 "description/basketball/nba?marketFilterId=def&liveOnly=true&lang=en",
  # #                 "https://www.bovada.lv/services/sports/event/v2/events/" 
  # #                 "A/description/basketball/nba?marketFilterId=def&preMatchOnly=true&lang=en"]
		# self.scores_url = "https://services.bovada.lv/services/sports/results/api/v1/scores/"
		# self.prev_states = []
		# self.cur_states = []
		# self.games = []  # parsed cur_states
		# self.events = []
		# self.jsons = []
		# self.get_cur_state()
		# self.parse()
        self.sip = 

	def next(self):

	def get_cur_state(self):
		self.jsons = []
		json = None
		for i, link in enumerate(self.links):
			while json is None:
				json = req(link)
			if json is not None:
				self.jsons.append(json)
	def parse(self):
		df = self.cur_states[0]
		df = df[0]
		live_games = df['events']
		self.events = []
		for live_game in live_games:
			game_id = live_game['id']
			h_team = live_game['competitors'][0]['name']
			a_team = live_game['competitors'][1]['name']
			a_ml = live_game['displayGroups'][0]['markets'][1]['outcomes'][0]['price']['american']
			h_ml = live_game['displayGroups'][0]['markets'][1]['outcomes'][1]['price']['american']
			a_spread = live_game['displayGroups'][0]['markets'][0]['outcomes'][0]['price']['handicap']
			h_spread = live_game['displayGroups'][0]['markets'][0]['outcomes'][1]['price']['handicap']
			a_odds_ps = live_game['displayGroups'][0]['markets'][0]['outcomes'][0]['price']['american']
			h_odds_ps = live_game['displayGroups'][0]['markets'][0]['outcomes'][1]['price']['american']
			u_odds_total = live_game['displayGroups'][0]['markets'][2]['outcomes'][1]['price']['american']
			o_odds_total = live_game['displayGroups'][0]['markets'][2]['outcomes'][0]['price']['american']
			total = live_game['displayGroups'][0]['markets'][2]['outcomes'][0]['price']['handicap']

			score_json = req(self.scores_url + game_id) #includes baskets and times for future sync but for now are just getting current score
			h_pts = score_json['latestScore']['home']
			a_pts = score_json['latestScore']['away']
			last_updated = score_json['lastUpdated']
			time = time.time()
			quarter = score_json['clock']['periodNumber']
			total = live_game[]

			# zero for things we are missing got to check over under since on nba2 anand sillily put home and away
			ordered_list = [0, 0, game_id, a_team, h_team, time, 0, 0, quarter, 0, a_pts, h_pts, 0, 0, 0, 0, 
							last_updated, 0, a_ml, h_ml, a_odds_ps, h_odds_ps, 0, 0, a_spread, h_spread, 
							u_odds_total, o_odds_total, 0, 0, total, total, 0] 
			self.events.append(ordered_list)




class SipEnv4:
	def __init__(self):
        self.scraper = BovadaState()
        self.cur_states = None

    def step(self, actions):
    	self.cur_states = self.scraper.next()  # list of game rows, dtype=tensor
    	self.actions = actions
    	rewards = []
    	reward = 0
    	game_ids = []
    	current_num = len(self.cur_states)
    	for cur_state in self.cur_states:
    		for i in range(current_num):
    			for action in self.actions:
    				game_id = action[1]
    				if game_id == cur_State[2]:
    					self._odds()
    					reward = self.act()
    					rewards += reward
    	return self.cur_states, rewards, done, self.odds
        
    		if cur_state[2] ==
    	self._odds()

    	rewards = []
    	for game in self.games:
    		rewards.append(self.act(action[0]))

    	return next_states, rewards, d, misc

    def act(self):
    	r = 0
    	if self.game_over == True:
    		for bet in self.game_bets:
    			if bet.team == winning_team:
    				reward = bet.winning_amt / bet.risk_amt
    				r += reward
    			if bet.team == losing_team:
    				reward = -1
    				r += reward

    	else:
    		if action[0] == skip:
    			r = 0 

    		else:
    			r = .01
    			self._bet()
		return r

	def _bet(self):
        # we don't update self.money because we don't want it to get a negative reward on _bet()
        print("bet*")
        bet = Bet(self.action, self.odds, self.cur_state)
        self.game_bets.append(bet)



def req(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
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


class Bet:
    # class storing bet info, will be stored in pair (hedged-bet)
    # might want to add time into game so we can easily aggregate when it is betting in the game
    # possibly using line numbers where they update -(1/(x-5)). x=5 is end of game
    # maybe bets should be stored as a step (csv line) and the bet amt and index into game.
    def __init__(self, action, odds, cur_state):
        self.amt = amt
        self.team = action  # 0 for away, 1 for home
        self.a_odds = odds[0]
        self.h_odds = odds[1]
        self.cur_state = cur_state
        self.wait_amt = 0

    def reset_odds(self):
        # reset both odds
        self.a_odds = 0
        self.h_odds = 0

    def __repr__(self):
        # simple console log of a bet
        print(h.act(self.team))
        print('bet amt: ' + str(self.amt) + ' | team: ' + str(self.team))
        print('a_odds: ' + str(self.a_odds) + ' | h_odds: ' + str(self.h_odds))
