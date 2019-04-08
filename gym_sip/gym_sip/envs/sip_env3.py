import requests 
import helpers as h

HEADERS = {'User-Agent': 'Mozilla/5.0'}

class BovadaState:
	def __init__(self):
		self.links = ["https://www.bovada.lv/services/sports/event/v2/events/A/" 
                  "description/basketball/nba?marketFilterId=def&liveOnly=true&lang=en",
                  "https://www.bovada.lv/services/sports/event/v2/events/" 
                  "A/description/basketball/nba?marketFilterId=def&preMatchOnly=true&lang=en"]
		self.scores_url = "https://services.bovada.lv/services/sports/results/api/v1/scores/"
		self.prev_states = []
		self.cur_states = []
		self.games = []  # parsed cur_states
		self.get_cur_state()
	def next(self):
		self.prev_states = self.cur_states
		self.get_cur_state()
		self.parse()
		return self.cur_states
	def get_cur_state(self):
		jsons = []
		json = None
		for i, link in enumerate(self.links):
			while json is None:
				json = req(link)
			if json is not None:
				self.cur_states[i] = json
	def parse(self):
		df = self.cur_states[0]
		df = df[0]
		live_games = df['events']
		for live_game in live_games:
			game_id = live_game['id']
			h_team = live_game['competitors'][0]['name']
			a_team = live_game['competitors'][1]['name']
			a_ml = live_game['displayGroups'][0]['markets'][1]['outcomes'][0]['price']['american']
			h_ml = live_game['displayGroups'][0]['markets'][1]['outcomes'][1]['price']['american']
			a_spread = live_game['displayGroups'][0]['markets'][0]['outcomes'][0]['price']['american']
			h_spread = live_game['displayGroups'][0]['markets'][0]['outcomes'][1]['price']['american']
			score_json = req(self.scores_url + game_id) #includes baskets and times for future sync but for now are just getting current score
			h_pts = score_json['latestScore']['home']
			a_pts = score_json['latestScore']['away']
			last_updated = score_json['lastUpdated']



class SipEnv3:
	def __init__(self):
        self.scraper = BovadaState()
        self.cur_states = None

    def step(self, action):
    	self.cur_states = scraper.next()  # list of game rows, dtype=tensor
    	rewards = []
    	for game in self.games:
    		rewards.append(self.act(action))

    	return next_states, rewards, d, misc

    def act(self):
    	r = 0
    	return r


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
