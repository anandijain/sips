# macros.py for highly usable data
url = 'https://www.baseball-reference.com'

bk_url = 'https://www.basketball-reference.com'

fb_url = 'https://www.pro-football-reference.com'

nhl_url = 'https://www.hockey-reference.com'

teams_url = '/teams/'

schedule_suffix = '-schedule-scores.shtml'

mlb_teams_short = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET',
                   'HOU', 'KCR', 'ANA', 'LAD', 'FLA', 'MIL', 'MIN', 'NYM', 'NYY', 'OAK',
                   'PHI', 'PIT', 'SDP', 'SFG', 'SEA', 'STL', 'TBD', 'TEX', 'TOR', 'WSN']

mlb_teams_full = ['Arizona Diamondbacks', 'Atlanta Braves', 'Baltimore Orioles',
                  'Boston Red Sox', 'Chicago Cubs', 'Chicago White Sox', 'Cincinnati Reds',
                  'Cleveland Indians', 'Colorado Rockies', 'Detroit Tigers', 'Houston Astros',
                  'Kansas City Royals', 'Los Angeles Angels', 'Los Angeles Dodgers',
                  'Miami Marlins', 'Milwaukee Brewers', 'Minnesota Twins', 'New York Mets',
                  'New York Yankees', 'Oakland Athletics', 'Philadelphia Phillies',
                  'Pittsburgh Pirates', 'San Diego Padres', 'San Francisco Giants',
                  'Seattle Mariners', 'St. Louis Cardinals', 'Tampa Bay Rays',
                  'Texas Rangers', 'Toronto Blue Jays', 'Washington Nationals']


mlb_csv_columns = ['game_id', 'inning', 'score', 'outs', 'robs',
                   'pitch_count', 'pitches', 'runs_outs', 'at_bat', 'wwpa', 'wwe', 'desc']


bb_ref_box_meta = ['id', 'date', 'start_time', 'attendance', 'venue', 'game_duration', 'game_type', 'ump_hp', 'ump_1b', 'ump_2b', 'ump_3b',
                   'start_weather']

bb_ref_lineup = ['id']
