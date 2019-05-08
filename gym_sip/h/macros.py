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


bovada_headers = ['sport', 'league', 'game_id', 'a_team', 'h_team', 'cur_time',
                'lms_date','lms_time','quarter','secs','a_pts','h_pts','status',
                'a_win','h_win','last_mod_to_start',
                'last_mod_lines','num_markets','a_ml','h_ml','a_deci_ml','h_deci_ml',
                'a_odds_ps','h_odds_ps','a_deci_ps','h_deci_ps','a_hcap_ps','h_hcap_ps','a_odds_tot',
                'h_odds_tot','a_deci_tot','h_deci_tot','a_hcap_tot','h_hcap_tot','game_start_time']

# bask
sports = ['basketball/nba?marketFilterId=def&liveOnly=true&lang=en',
'basketball/nba?marketFilterId=def&preMatchOnly=true&eventsLimit=10&lang=en',
# baseball
'baseball/mlb?marketFilterId=def&liveOnly=true&lang=en',
'baseball/mlb?marketFilterId=def&preMatchOnly=true&eventsLimit=10&lang=en',
# college bball
'basketball/college-basketball?marketFilterId=def&liveOnly=true&lang=en',
'basketball/college-basketball?marketFilterId=def&preMatchOnly=true&eventsLimit=10&lang=en',
# esports
'esports?marketFilterId=def&liveOnly=true&lang=en',
'esports?marketFilterId=def&preMatchOnly=true&eventsLimit=10&lang=en',
# football
'football?marketFilterId=def&liveOnly=true&lang=en',
'football?marketFilterId=def&preMatchOnly=true&eventsLimit=10&lang=en',
# tennis
'tennis?marketFilterId=def&liveOnly=true&lang=en',
'tennis?marketFilterId=def&preMatchOnly=true&eventsLimit=10&lang=en',
# volley
'volleyball?marketFilterId=def&liveOnly=true&lang=en',
'volleyball?marketFilterId=def&preMatchOnly=true&eventsLimit=10&lang=en']
