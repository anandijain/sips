
from sips.macros import sports_ref as sref
from sips.h import grab
from sips.h import parse

comment_idxs = {
    25: 'per_minute',
    26: 'per_poss',
    30: 'advanced',
    31: 'shooting',
    32: 'pbp',
    33: 'year-and-career-highs',
    34: 'playoffs_per_game',
    35: 'playoffs_totals',
    36: 'playoffs_per_minute',
    37: 'playoffs_per_poss',
    38: 'playoffs_advanced',
    39: 'playoffs_shooting',
    40: 'playoffs_pbp',
    41: 'year-and-career-highs-po',
    42: 'all_star',
    43: 'sim_thru',
    44: 'sim_career'
}


def player(player_sfx):
    url = sref.bk_url + player_sfx
    table_ids = ['per_game', 'totals']
    dfs = []
    p = grab.get_page(url)
    for t_id in table_ids:
        dfs.append(parse.get_table(p, t_id, to_pd=True))

    cs = parse.comments(p)
    for index, t_id in comment_idxs.items():
        soup = parse.to_soup(cs[index])
        df = parse.get_table(soup, t_id, to_pd=True)
        dfs.append(df)
    return dfs


if __name__ == "__main__":
    sfx = '/players/j/jamesle01.html'
    lebron = player(sfx)
    print(lebron)