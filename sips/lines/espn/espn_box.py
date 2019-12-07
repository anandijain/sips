"""

"""
import time
from sips.h import grab as g

ESPN_ROOT = "https://www.espn.com/"


TIME_GAME_TUP = ("a", "name", "&lpos=nfl:schedule:time")
TIME_GAME_TUP = ("a", "name", "&lpos=nfl:schedule:score")
TIME_GAME_TUP = ("a", "name", "&lpos=nfl:schedule:live")


def get_sports():
    sports = [
        "football/nfl",
        "baseball/mlb",
        "basketball/nba",
        "football/college-football",
        "basketball/mens-college-basketball",
    ]
    return sports


def time_ids(page=None, sport="football/nfl"):
    if not page:
        page = schedule(sport)
    unparsed_ids = page.find_all("a", {"name": "&lpos=" + sport + ":schedule:time"})
    return unparsed_ids


def score_ids(page=None, sport="football/nfl"):
    if not page:
        page = schedule(sport)
    unparsed_ids = page.find_all("a", {"name": "&lpos=" + sport + ":schedule:score"})
    return unparsed_ids


def live_ids(page=None, sport="football/nfl"):
    if not page:
        page = schedule(sport)
    unparsed_ids = page.find_all("a", {"name": "&lpos=" + sport + ":schedule:live"})
    live_ids = parse_live_ids(unparsed_ids)
    return live_ids


def get_all_ids(page=None, sports=get_sports()):
    ids = {}
    for sport in sports:
        ids[sport] = get_ids(sport)
    return ids


def live_links(page=None, sport="football/nfl"):
    if not page:
        page = schedule(sport)
    ids = live_ids(page, sport)
    links = boxlinks(ids, sport)
    return links


def schedules(sports=["football/nfl"]):
    if not sports:
        sports = get_sports()

    pages = []
    for sport in sports:
        espn_id_link = "/" + sport + "/schedule"
        p = g.page(espn_id_link)
        pages.append(p)
    return pages


def schedule(sport="football/nfl"):
    espn_id_link = ESPN_ROOT + sport + "/schedule"
    p = g.page(espn_id_link)
    return p


def get_ids(sport="football/nfl", live_only=True):
    """

    """
    p = schedule(sport)
    ids_live = live_ids(p, sport)
    if live_only:
        return ids_live
    ids_score = score_ids(p, sport)
    ids_time = time_ids(p, sport)
    ids_parsed = parse_ids(ids_score + ids_time)
    ids = ids_live + ids_parsed
    return ids


def get_live_pages():
    ids = live_ids()
    pages = []
    # add multithreading
    for id in ids:
        pages.append(g.page(id_to_boxlink(id)))
    return pages


def get_live_boxes(pages=None):
    if not pages:
        pages = get_live_pages()
    boxes = []
    for p in pages:
        boxes.append(boxscore(p))
    return boxes


def parse_live_id(tag):
    # print(tag)
    id = tag["href"].split("=")[-1]
    return id


def parse_live_ids(tags):
    ret = []
    if isinstance(tags, dict):
        for v in tags.values():
            v = parse_live_id(v)
        return ret
    else:
        for tag in tags:
            ret.append(parse_live_id(tag))
    return ret


def tag_to_id(tag):
    id = tag["href"].split("/")[-1]
    return id


def parse_id_dict(tags_dict):
    for k, v in tags_dict.items():
        tags_dict[k] = [parse_live_id(tag) for tag in v]
    return tags_dict


def parse_ids(tags):
    """

    """
    parsed_ids = []
    if tags:
        if isinstance(tags, dict):
            return parse_id_dict(tags)
        for tag in tags:
            try:
                game_id = tag_to_id(tag)
            except TypeError:
                pass
            try:
                game_id = game_id.split("=")[-1]
            except TypeError:
                pass
            parsed_ids.append(game_id)
    else:
        return None
    return parsed_ids


def id_to_boxlink(id, sport="football/nfl"):
    return ESPN_ROOT + sport + "/boxscore?gameId=" + id


def box_tds(tds):
    x = None
    data = []
    for td in tds:
        if x == 1:
            data.append(td.text)
        txt = td.text
        if txt == "TEAM":
            x = 1
    return data


def teamstats(page):
    """

    """
    a_newstats = []
    h_newstats = []
    # if table_index % 2 == 1, then home_team
    tables = page.find_all("div", {"class": "content desktop"})
    for i, table in enumerate(tables):
        header = table.find("thead")
        len_header = len(header.find_all("th")) - 1  # - 1 for 'TEAM'
        tds = table.find_all("td")
        data = box_tds(tds)
        if i % 2 == 1:
            if len(data) == 0:
                h_newstat = ["NaN" for _ in range(len_header)]
            else:
                h_newstat = data
            h_newstats.append(h_newstat)
        else:
            if len(data) == 0:
                a_newstat = ["NaN" for _ in range(len_header)]
            else:
                a_newstat = data
            a_newstats.append(a_newstat)
    return a_newstats, h_newstats


def parse_teamstats(teamstats):
    """
    a @ h order
    """
    real_stats = []
    for team_newstats in teamstats:
        for team_newstat in team_newstats:
            for stat in team_newstat:
                try:
                    real_stat = stat.text
                    if real_stat == "TEAM":
                        continue
                except AttributeError:
                    real_stat = stat
                real_stats.append(real_stat)
    return real_stats


def box_teamnames(page):
    """
    A @ H always
    """
    teams = page.find_all("span", {"class": "short-name"})
    destinations = page.find_all("span", {"class": "long-name"})
    names = [team.text for team in teams]
    cities = [destination.text for destination in destinations]
    if not names or not cities:
        return None
    a_team, h_team = [dest + " " + name for (dest, name) in zip(cities, names)]
    return a_team, h_team


def boxlinks(ids=None, sports=["football/nfl"], live_only=True):
    """

    """
    links = []
    for sport in sports:
        if not ids:
            ids = get_ids(sport=sport, live_only=live_only)
        url = ESPN_ROOT + sport + "/boxscore?gameId="
        sport_links = [url + id for id in ids]
        links += sport_links
    return links


def boxscores(sports=["football/nfl"], output="dict"):
    """
    ~ 10 seconds
    """
    links = boxlinks(sports=sports)
    # print(f'links: {links}')
    boxes = [boxscore(link) for link in links]
    return boxes


def boxscore(link):
    """

    """
    page = g.page(link)
    teams = box_teamnames(page)
    if teams:
        a_team, h_team = teams
    else:
        a_team, h_team = None, None
    team_stats = teamstats(page)
    real_stats = parse_teamstats(team_stats)
    real_stats.extend([a_team, h_team])
    return real_stats


def main():
    enum = enumerate
    start = time.time()

    real_stats = boxscores()
    for i, rs in enum(real_stats):
        print(f"{i}: {len(rs)}")
        print(rs)
    print(real_stats)
    end = time.time()
    delta = end - start
    print(f"delta: {delta}")
    return real_stats


if __name__ == "__main__":
    main()
