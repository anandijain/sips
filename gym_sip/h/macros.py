# Heavily referenced data
link = "https://www.bovada.lv/services/sports/event/v2/events/A/description/"

az_teams = ['Atlanta Hawks', 'Boston Celtics',
            'Brooklyn Nets', 'Charlotte Hornets',
            'Chicago Bulls', 'Cleveland Cavaliers',
            'Dallas Mavericks', 'Denver Nuggets',
            'Detroit Pistons', 'Golden State Warriors',
            'Houston Rockets', 'Indiana Pacers',
            'Los Angeles Clippers', 'Los Angeles Lakers',
            'Memphis Grizzlies', 'Miami Heat',
            'Milwaukee Bucks', 'Minnesota Timberwolves',
            'New Orleans Pelicans', 'New York Knicks',
            'Oklahoma City Thunder', 'Orlando Magic',
            'Philadelphia 76ers', 'Phoenix Suns',
            'Portland Trail Blazers', 'Sacramento Kings',
            'San Antonio Spurs', 'Toronto Raptors',
            'Utah Jazz', 'Washington Wizards']


nba2_headers = ['a_team', 'h_team', 'sport', 'league', 'game_id', 'cur_time',
                'a_pts', 'h_pts', 'secs', 'status', 'a_win', 'h_win', 'last_mod_to_start', 'last_mod_lines',
                'num_markets', 'a_ml', 'h_ml', 'a_hcap_tot', 'h_hcap_tot', 'game_start_time']
"""    
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
        elif league == 8:
            self.links = ["https://www.bovada.lv/services/sports/event/v2/events/A/description/basketball/nba?marketFilterId=rank&liveOnly=true&lang=en",
                          "https://www.bovada.lv/services/sports/event/v2/events/A/description/basketball/nba?marketFilterId=rank&preMatchOnly=true&lang=en"]
        else:               # All BASK
            self.links = ["https://www.bovada.lv/services/sports/event/v2/events/A/" 
                          "description/basketball?marketFilterId=def&liveOnly=true&eventsLimit=8&lang=en",
                          "https://www.bovada.lv/services/sports/event/v2/events/A/" 
                          "description/basketball?marketFilterId=def&preMatchOnly=true&eventsLimit=50&lang=en"]
"""
