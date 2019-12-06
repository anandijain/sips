"""

"""
import sips.h.grab as g
from sips.macros import bov as bm
from sips.lines.bov.utils import bov_utils as u


def get_scores(events, session=None):
    """
    {game_id : quarter, secs, a_pts, h_pts, status}

    """
    ids = u.get_ids(events)
    links = [bm.BOV_SCORES_URL + game_id for game_id in ids]
    if session:
        raw = g.async_req(links, output="dict", key="eventId", session=session)
    else:
        raw = g.reqs_json(links)
    scores_dict = {g_id: score(j) for g_id, j in raw.items()}
    return scores_dict


def score(json_data):
    """
    given json data for a game_id, returns the score data of the game
    
    """
    [quarter, secs, a_pts, h_pts, game_status] = [None for _ in range(5)]

    if not json_data:
        return [quarter, secs, a_pts, h_pts, game_status]

    clock = json_data.get("clock")
    if clock:
        quarter = clock["periodNumber"]
        secs = clock["relativeGameTimeInSecs"]
    latest_score = json_data.get("latestScore")

    if not latest_score:
        a_pts = 0
        h_pts = 0
    else:
        a_pts = latest_score.get("visitor")
        h_pts = latest_score.get("home")

    status = json_data.get("gameStatus")
    if status:
        game_status = status
    else:
        status = None
    return [quarter, secs, a_pts, h_pts, game_status]
