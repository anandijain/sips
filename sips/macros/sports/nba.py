teams = [
    "Atlanta Hawks",
    "Boston Celtics",
    "Brooklyn Nets",
    "Charlotte Hornets",
    "Chicago Bulls",
    "Cleveland Cavaliers",
    "Dallas Mavericks",
    "Denver Nuggets",
    "Detroit Pistons",
    "Golden State Warriors",
    "Houston Rockets",
    "Indiana Pacers",
    "Los Angeles Clippers",
    "Los Angeles Lakers",
    "Memphis Grizzlies",
    "Miami Heat",
    "Milwaukee Bucks",
    "Minnesota Timberwolves",
    "New Jersey Nets",
    "New Orleans Pelicans",
    "New York Knicks",
    "Oklahoma City Thunder",
    "Orlando Magic",
    "Philadelphia 76ers",
    "Phoenix Suns",
    "Portland Trail Blazers",
    "Sacramento Kings",
    "San Antonio Spurs",
    "Seattle SuperSonics",
    "Toronto Raptors",
    "Utah Jazz",
    "Vancouver Grizzlies",
    "Washington Wizards",
]


abrvs = [
    "ATL",
    "BKN",
    "BOS",
    "CHA",
    "CHI",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GSW",
    "HOU",
    "IND",
    "LAC",
    "LAL",
    "MEM",
    "MIA",
    "MIL",
    "MIN",
    "NOP",
    "NYK",
    "OKC",
    "ORL",
    "PHI",
    "PHX",
    "POR",
    "SAC",
    "SAS",
    "TOR",
    "UTA",
    "WAS",
]

index = {
    """
    year/career highs
    totals
    per min
    per game
    advanced
    sim thru

    sim career
    shooting
    pbp
    all salaries
    college stats


    """
}

# bk-ref player columns


PLAYER_TOTALS = ['index', 'Season', 'Age', 'Tm', 'Lg', 'Pos', 'G', 'GS', 'MP', 'FG',
                 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT',
                 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
                 'PTS']

PLAYER_YEAR_CAREER_HIGHS_PO = ['index', 'Season', 'Age',  'Tm',  'Lg',  'MP',  'FG',
                               'FGA',  'FT', 'FTA', 'TRB', 'AST',  'PF', 'PTS']

# drop 1
PLAYER_YEAR_CAREER_HIGHS = ['index',
                            'Season',
                            'Age',
                            'Tm',
                            'Lg',
                            'highs_MP',
                            'highs_FG',
                            'highs_FGA',
                            'highs_3P',
                            'highs_3PA',
                            'highs_2P',
                            'highs_2PA',
                            'highs_FT',
                            'highs_FTA',
                            'highs_ORB',
                            'highs_DRB',
                            'highs_TRB',
                            'highs_AST',
                            'highs_STL',
                            'highs_BLK',
                            'highs_TOV',
                            'highs_PF',
                            'highs_PTS',
                            'highs_GmSc']


PLAYER_PER_MIN = ['index', 'Season', 'Age', 'Tm', 'Lg', 'Pos', 'G', 'GS', 'MP', 'FG',
                  'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'FT', 'FTA',
                  'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

PLAYER_PER_GAME = ['index', 'Season', 'Age', 'Tm', 'Lg', 'Pos', 'G', 'GS', 'MP', 'FG',
                   'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT',
                   'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
                   'PTS']

PLAYER_ADVANCED = ['index', 'Season', 'Age', 'Tm', 'Lg', 'Pos', 'G', 'MP', 'PER',
                   'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%',
                   'TOV%', 'USG%', 'Unnamed: 19', 'OWS', 'DWS', 'WS', 'WS/48',
                   'Unnamed: 24', 'OBPM', 'DBPM', 'BPM', 'VORP']

PLAYER_SALARIES = ['index', 'Season', 'Team', 'Lg', 'Salary']

# 2 level, drop a row
PLAYER_COLLEGE = ['index'
                  'Season',
                  'Age',
                  'College',
                  'Totals', 'G',
                  'Totals_MP',
                  'Totals_FG',
                  'Totals_FGA',
                  'Totals_3P',
                  'Totals_3PA',
                  'Totals_FT',
                  'Totals_FTA',
                  'Totals_ORB',
                  'Totals_TRB',
                  'Totals_AST',
                  'Totals_STL',
                  'Totals_BLK',
                  'Totals_TOV',
                  'Totals_PF',
                  'Totals_PTS',
                  'Shooting_FG%',
                  'Shooting_3P%',
                  'Shooting_FT%',
                  'Per Game_MP',
                  'Per Game_PTS',
                  'Per Game_TRB',
                  'Per Game_AST']


# 3 level index so drop 2 rows
PLAYER_SHOOTING = ['index',
                   'Season',
                   'Age',
                   'Tm',
                   'Lg',
                   'Pos',
                   'G',
                   'MP',
                   'FG%',
                   'Dist.',
                   'pct_FGA_by_Distance_2P',
                   'pct_FGA_by_Distance_0-3',
                   'pct_FGA_by_Distance_3-10',
                   'pct_FGA_by_Distance_10-16',
                   'pct_FGA_by_Distance_16-3pt',
                   'pct_FGA_by_Distance_3P',
                   'FG_pct_by_Distance_2P',
                   'FG_pct_by_Distance_0-3',
                   'FG_pct_by_Distance_3-10',
                   'FG_pct_by_Distance_10-16',
                   'FG_pct_by_Distance_16-3pt',
                   'FG_pct_by_Distance_3P',
                   '2-Pt_Field_Goals_pct_astd',
                   '2-Pt_Field_Goals_Dunks_pct_FGA',
                   '2-Pt_Field_Goals.2_Dunks_Md.',
                   '3-Pt_Field_Goals_pct_astd',
                   '3-Pt_Field_Goals_Corner_pct_3PA',
                   '3-Pt_Field_Goals_Corner_3P_pct',
                   '3-Pt_Field_Goals_Heaves_Att.',
                   '3-Pt_Field_Goals_Heaves_Md.']

# drop 1 row
PLAYER_PBP = ['index',
              'Season',
              'Age',
              'Tm',
              'Lg',
              'Pos',
              'G',
              'MP',
              'pos_est_PG%',
              'pos_est_SG%',
              'pos_est_SF%',
              'pos_est_PF%',
              'pos_est_C%',
              '+/- Per 100 Poss_OnCourt',
              '+/- Per 100 Poss_On-Off',
              'Turnovers_BadPass',
              'Turnovers_1_LostBall',
              'Fouls Committed_Shoot',
              'Fouls Committed_1_Off.',
              'Fouls Drawn_Shoot',
              'Fouls Drawn_1_Off',
              'Misc_PGA',
              'Misc_And1',
              'Misc_Blkd']

# drop one row
PLAYER_SIM_THRU = ['index', 'Thru 10 Years', 'Thru 10 Years.1',
                   'best_to_worst', 'best_to_worst_1',
                   'best_to_worst_2', 'best_to_worst_3',
                   'best_to_worst_4', 'best_to_worst_5',
                   'best_to_worst_6', 'best_to_worst_7',
                   'best_to_worst_8', 'best_to_worst_9']


# drop one row
PLAYER_SIM_CAREER = ['index', 'Career', 'Career.1', 'best_to_worst',
                     'best_to_worst_1', 'best_to_worst_2',
                     'best_to_worst_3', 'best_to_worst_4',
                     'best_to_worst_5', 'best_to_worst_6',
                     'best_to_worst_7', 'best_to_worst_8',
                     'best_to_worst_9', 'best_to_worst_10',
                     'best_to_worst_11', 'best_to_worst_12',
                     'best_to_worst_13', 'best_to_worst_14',
                     'best_to_worst_15', 'best_to_worst_16']


PLAYER_TABLES = {
    'totals': PLAYER_TOTALS,
    'year-and-career-highs-po': PLAYER_YEAR_CAREER_HIGHS_PO,
    'year-and-career-highs': PLAYER_YEAR_CAREER_HIGHS,
    'per_minute': PLAYER_PER_MIN,
    'per_game': PLAYER_PER_GAME,
    'advanced': PLAYER_ADVANCED,
    'all_salaries': PLAYER_SALARIES,
    'all_college_stats': PLAYER_COLLEGE,
    'shooting': PLAYER_SHOOTING,
    'pbp': PLAYER_PBP,
    'sim_thru': PLAYER_SIM_THRU,
    'sim_career': PLAYER_SIM_CAREER,
}

PLAYER_DROP_N = {
    'totals': 0,
    'year-and-career-highs-po': 0,
    'year-and-career-highs': 1,
    'per_minute':  0,
    'per_game': 0,
    'advanced':  0,
    'all_salaries':  0,
    'all_college_stats':  1,
    'shooting':  2,
    'pbp':  1,
    'sim_thru':  1,
    'sim_career':  1,
}


# bk-ref boxscore columns

LINE_SCORES = ["win", "team", "q_1", "q_2", "q_3", "q_4", "T"]

FOUR_FACTORS = ["win", "team", "Pace",
                "eFG%", "TOV%", "ORB%", "FT/FGA", "ORtg"]

GAME_BASIC = [
    "index",
    "Starters",
    "MP",
    "FG",
    "FGA",
    "FG%",
    "3P",
    "3PA",
    "3P%",
    "FT",
    "FTA",
    "FT%",
    "ORB",
    "DRB",
    "TRB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
    "+/-",
]

GAME_ADVANCED = [
    "index",
    "Starters",
    "MP",
    "TS%",
    "eFG%",
    "3PAr",
    "FTr",
    "ORB%",
    "DRB%",
    "TRB%",
    "AST%",
    "STL%",
    "BLK%",
    "TOV%",
    "USG%",
    "ORtg",
    "DRtg",
]

GAME_PBP = [
    'index',
    'Time',
    'a_team_desc',
    'a_pt_change',
    'score',
    'h_pt_change',
    'h_team_desc',
]


# TEAM SET COLS FROM HERE DOWN
POST_GAME = [
    "A_1stq",
    "A_2ndq",
    "A_3_point_attempt_rate",
    "A_3_point_percentage",
    "A_3_pointers_attempted",
    "A_3_pointers_made",
    "A_3rdq",
    "A_4thq",
    "A_assist_percentage",
    "A_assists",
    "A_block_percentage",
    "A_blocks",
    "A_defensive_rating",
    "A_defensive_rebound_percentage",
    "A_defensive_rebounds",
    "A_effective_field_goal_percentage",
    "A_efg%",
    "A_field_goal_percentage",
    "A_field_goals_attempted",
    "A_field_goals_made",
    "A_free_throw_percentage",
    "A_free_throw_rate",
    "A_free_throws_attempted",
    "A_free_throws_made",
    "A_ft/fga",
    "A_minutes_played",
    "A_minutes_played_2",
    "A_offensive_rating",
    "A_offensive_rebound_percentage",
    "A_offensive_rebounds",
    "A_orb%",
    "A_ortg",
    "A_ot1",
    "A_ot2",
    "A_ot3",
    "A_ot4",
    "A_pace",
    "A_personal_fouls",
    "A_plus_minus",
    "A_points",
    "A_steal_percentage",
    "A_steals",
    "A_team",
    "A_total",
    "A_total_rebound_percentage",
    "A_total_rebounds",
    "A_tov%",
    "A_true_shooting_percentage",
    "A_turnover_percentage",
    "A_turnovers",
    "A_usage_percentage",
    "A_win",
    "Arena",
    "Date",
    "Game_id",
    "H_1stq",
    "H_2ndq",
    "H_3_point_attempt_rate",
    "H_3_point_percentage",
    "H_3_pointers_attempted",
    "H_3_pointers_made",
    "H_3rdq",
    "H_4thq",
    "H_assist_percentage",
    "H_assists",
    "H_block_percentage",
    "H_blocks",
    "H_defensive_rating",
    "H_defensive_rebound_percentage",
    "H_defensive_rebounds",
    "H_effective_field_goal_percentage",
    "H_efg%",
    "H_field_goal_percentage",
    "H_field_goals_attempted",
    "H_field_goals_made",
    "H_free_throw_percentage",
    "H_free_throw_rate",
    "H_free_throws_attempted",
    "H_free_throws_made",
    "H_ft/fga",
    "H_minutes_played",
    "H_minutes_played_2",
    "H_offensive_rating",
    "H_offensive_rebound_percentage",
    "H_offensive_rebounds",
    "H_orb%",
    "H_ortg",
    "H_ot1",
    "H_ot2",
    "H_ot3",
    "H_ot4",
    "H_pace",
    "H_personal_fouls",
    "H_plus_minus",
    "H_points",
    "H_steal_percentage",
    "H_steals",
    "H_team",
    "H_total",
    "H_total_rebound_percentage",
    "H_total_rebounds",
    "H_tov%",
    "H_true_shooting_percentage",
    "H_turnover_percentage",
    "H_turnovers",
    "H_usage_percentage",
    "H_win",
    "Season",
    "Time",
]

HISTORY_NOML = [
    "A_team_x",
    "Away_team_gen_avg_3_point_attempt_rate",
    "Away_team_gen_avg_3_pointers_attempted",
    "Away_team_gen_avg_3_pointers_made",
    "Away_team_gen_avg_assist_percentage",
    "Away_team_gen_avg_assists",
    "Away_team_gen_avg_block_percentage",
    "Away_team_gen_avg_blocks",
    "Away_team_gen_avg_defensive_rating",
    "Away_team_gen_avg_defensive_rebound_percentage",
    "Away_team_gen_avg_defensive_rebounds",
    "Away_team_gen_avg_effective_field_goal_percentage",
    "Away_team_gen_avg_field_goal_percentage",
    "Away_team_gen_avg_field_goals_attempted",
    "Away_team_gen_avg_field_goals_made",
    "Away_team_gen_avg_free_throw_percentage",
    "Away_team_gen_avg_free_throw_rate",
    "Away_team_gen_avg_free_throws_attempted",
    "Away_team_gen_avg_free_throws_made",
    "Away_team_gen_avg_ft/fga",
    "Away_team_gen_avg_offensive_rating",
    "Away_team_gen_avg_offensive_rebound_percentage",
    "Away_team_gen_avg_offensive_rebounds",
    "Away_team_gen_avg_op_ft/fga",
    "Away_team_gen_avg_pace",
    "Away_team_gen_avg_score",
    "Away_team_gen_avg_score_allowed",
    "Away_team_gen_avg_steal_percentage",
    "Away_team_gen_avg_steals",
    "Away_team_gen_avg_total_rebounds",
    "Away_team_gen_avg_true_shooting_percentage",
    "Away_team_gen_avg_turnover_percentage",
    "Away_team_gen_avg_turnovers",
    "Away_team_gen_avg_usage_percentage",
    "Away_team_gen_op_avg_3_point_attempt_rate",
    "Away_team_gen_op_avg_3_pointers_attempted",
    "Away_team_gen_op_avg_3_pointers_made",
    "Away_team_gen_op_avg_assist_percentage",
    "Away_team_gen_op_avg_assists",
    "Away_team_gen_op_avg_block_percentage",
    "Away_team_gen_op_avg_blocks",
    "Away_team_gen_op_avg_defensive_rating",
    "Away_team_gen_op_avg_defensive_rebound_percentage",
    "Away_team_gen_op_avg_defensive_rebounds",
    "Away_team_gen_op_avg_effective_field_goal_percentage",
    "Away_team_gen_op_avg_field_goal_percentage",
    "Away_team_gen_op_avg_field_goals_attempted",
    "Away_team_gen_op_avg_field_goals_made",
    "Away_team_gen_op_avg_free_throw_percentage",
    "Away_team_gen_op_avg_free_throw_rate",
    "Away_team_gen_op_avg_free_throws_attempted",
    "Away_team_gen_op_avg_free_throws_made",
    "Away_team_gen_op_avg_offensive_rating",
    "Away_team_gen_op_avg_offensive_rebound_percentage",
    "Away_team_gen_op_avg_offensive_rebounds",
    "Away_team_gen_op_avg_pace",
    "Away_team_gen_op_avg_steal_percentage",
    "Away_team_gen_op_avg_steals",
    "Away_team_gen_op_avg_total_rebound_percentage",
    "Away_team_gen_op_avg_total_rebounds",
    "Away_team_gen_op_avg_turnover_percentage",
    "Away_team_gen_op_avg_turnovers",
    "Date_x",
    "Game_id",
    "H_team_x",
    "Home_team_gen_avg_3_point_attempt_rate",
    "Home_team_gen_avg_3_pointers_attempted",
    "Home_team_gen_avg_3_pointers_made",
    "Home_team_gen_avg_assist_percentage",
    "Home_team_gen_avg_assists",
    "Home_team_gen_avg_block_percentage",
    "Home_team_gen_avg_blocks",
    "Home_team_gen_avg_defensive_rating",
    "Home_team_gen_avg_defensive_rebound_percentage",
    "Home_team_gen_avg_defensive_rebounds",
    "Home_team_gen_avg_effective_field_goal_percentage",
    "Home_team_gen_avg_field_goal_percentage",
    "Home_team_gen_avg_field_goals_attempted",
    "Home_team_gen_avg_field_goals_made",
    "Home_team_gen_avg_free_throw_percentage",
    "Home_team_gen_avg_free_throw_rate",
    "Home_team_gen_avg_free_throws_attempted",
    "Home_team_gen_avg_free_throws_made",
    "Home_team_gen_avg_ft/fga",
    "Home_team_gen_avg_offensive_rating",
    "Home_team_gen_avg_offensive_rebound_percentage",
    "Home_team_gen_avg_offensive_rebounds",
    "Home_team_gen_avg_op_ft/fga",
    "Home_team_gen_avg_pace",
    "Home_team_gen_avg_score",
    "Home_team_gen_avg_score_allowed",
    "Home_team_gen_avg_steal_percentage",
    "Home_team_gen_avg_steals",
    "Home_team_gen_avg_total_rebounds",
    "Home_team_gen_avg_true_shooting_percentage",
    "Home_team_gen_avg_turnover_percentage",
    "Home_team_gen_avg_turnovers",
    "Home_team_gen_avg_usage_percentage",
    "Home_team_gen_op_avg_3_point_attempt_rate",
    "Home_team_gen_op_avg_3_pointers_attempted",
    "Home_team_gen_op_avg_3_pointers_made",
    "Home_team_gen_op_avg_assist_percentage",
    "Home_team_gen_op_avg_assists",
    "Home_team_gen_op_avg_block_percentage",
    "Home_team_gen_op_avg_blocks",
    "Home_team_gen_op_avg_defensive_rating",
    "Home_team_gen_op_avg_defensive_rebound_percentage",
    "Home_team_gen_op_avg_defensive_rebounds",
    "Home_team_gen_op_avg_effective_field_goal_percentage",
    "Home_team_gen_op_avg_field_goal_percentage",
    "Home_team_gen_op_avg_field_goals_attempted",
    "Home_team_gen_op_avg_field_goals_made",
    "Home_team_gen_op_avg_free_throw_percentage",
    "Home_team_gen_op_avg_free_throw_rate",
    "Home_team_gen_op_avg_free_throws_attempted",
    "Home_team_gen_op_avg_free_throws_made",
    "Home_team_gen_op_avg_offensive_rating",
    "Home_team_gen_op_avg_offensive_rebound_percentage",
    "Home_team_gen_op_avg_offensive_rebounds",
    "Home_team_gen_op_avg_pace",
    "Home_team_gen_op_avg_steal_percentage",
    "Home_team_gen_op_avg_steals",
    "Home_team_gen_op_avg_total_rebound_percentage",
    "Home_team_gen_op_avg_total_rebounds",
    "Home_team_gen_op_avg_turnover_percentage",
    "Home_team_gen_op_avg_turnovers",
    "Season_x",
    "a_elo_pre",
    "h_elo_pre",
]
