from sips.macros import sports_ref as sref 
from sips.h import grab
from sips.h import parse

def get_links():
    p = grab.get_page(sref.bk_url + '/teams/')
    t = p.find("table", {"id": "teams_active"})
    links = [sref.bk_url + link['href'] for link in t.find_all("a")]
    return links

def get_teams():
    team_links = get_links()
    team_pages = grab.get_pages(team_links)
    return team_pages


if __name__ == "__main__":
    get_teams()