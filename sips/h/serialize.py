import numpy as np


def serialize_row(row, teams_dict, statuses_dict, include_teams=False):
    """
    going to take in something like this:
    ['FOOT', 5741304, 'Pittsburgh Steelers', 'Cleveland Browns', 1573540736617, 28,
    False, '0', '-1', '0', '0', 'PRE_GAME', '2.5', '-2.5', '-105', '-115', '+125',
    '-145', '40.0', '40.0', '-110', '-110', 'O', 'U', 1573780800000]
    and return a np array

    # note:
    serialize row needs to be refactored, i'm thinking that it should take in 
    a dataframe row and maybe have the option of what columns we want.
    so it can maximally serialize or, given a subset of columns, only serialize those 
    """
    ret = []
    row = list(row)
    teams = row[2:4]

    if include_teams:
        for t in teams:
            hot_teams = teams_dict[t]
            ret += hot_teams

    ret += row[4:6]

    if row[6]:
        ret += [1, 0]
    else:
        ret += [0, 1]

    ret += [row_ml(ml) for ml in row[7:11]]

    row_status = row[11]
    hot_status = statuses_dict[row_status]
    ret += hot_status
    mls = [row_ml(ml) for ml in row[12:22]]
    ret += mls
    final = np.array(ret, dtype=np.float32)
    return final


def row_ml(ml):
    """
    given a list of unparsed moneylines (eg can be 'EVEN' and None)
    edit the values such that 'EVEN' -> 100 and None -> -1
    typical order of list is [a0, h0, a1, h1]
    """
    if ml == "EVEN":
        ret = 100
    elif ml == None:
        ret = -1
    else:
        try:
            ret = float(ml)
        except:
            ret = -1
    return ret
