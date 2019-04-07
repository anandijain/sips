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
		self.games = None


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
