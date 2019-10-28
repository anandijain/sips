# Heavily referenced data
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


bov_game_info = ['sport', 'league', 'game_id', 'a_team', 'h_team', 'cur_time']


bov_score_headers = ['lms_date', 'lms_time', 'quarter', 'secs', 'a_pts',
                     'h_pts', 'status', 'a_win', 'h_win']


bov_lines_headers = ['last_mod_to_start', 'last_mod_lines', 'num_markets',
                     'a_ml', 'h_ml', 'a_deci_ml', 'h_deci_ml', 'a_odds_ps',
                     'h_odds_ps', 'a_deci_ps', 'h_deci_ps', 'a_hcap_ps',
                     'h_hcap_ps', 'a_odds_tot', 'h_odds_tot', 'a_deci_tot',
                     'h_deci_tot', 'a_hcap_tot', 'h_hcap_tot', 'a_ou', 'h_ou',
                     'game_start_time']


bovada_headers = bov_game_info + bov_score_headers + bov_lines_headers


BOV_URL = 'https://www.bovada.lv/services/sports/\
            event/v2/events/A/description/'
            
BOV_SCORES_URL = 'https://services.bovada.lv/services/\
                sports/results/api/v1/scores/'


suffixes = ['?marketFilterId=def&liveOnly=true&lang=en',
            '?marketFilterId=def&preMatchOnly=true&eventsLimit=10&lang=en']


sports = ['basketball', 'baseball', 'esports', 'football', 'football', 'tennis',
          'volleyball', 'hockey', 'badminton', 'table-tennis']


SPORT_TO_SUFFIX = {
    'nfl': 'football/nfl',
    'college-football': 'football/college-football',
    'nba': 'basketball/nba',
    'mlb': 'baseball/mlb'
}


def build_urls(sports=sports):
    '''
    returns list of urls
    '''
    urls = []
    for sport in sports:
        for suffix in suffixes:
            urls.append(BOV_URL + sport + suffix)
    return urls


def build_url_dict(sports=sports):
    '''
    returns dictionary of {
        sport (str) : url (str)
    }
    '''
    urls = {}
    for sport in sports:
        urls[sport] = [BOV_URL + sport + suffix for suffix in suffixes]
    return urls
