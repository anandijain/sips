from sips.macros import sports_ref as sref
from sips.h import grab
from sips.h import parse

table_ids = [
    "coaching_results",
    "coaching_ranks",
    "coaching_history",
    "challenge_results",
    "worked_for",
    "employed",
]


def coaches_summary(to_pd=True):
    coaches = grab.get_table(sref.NFL_URL + "coaches/", ["coaches"], to_pd=to_pd)
    return coaches


def coach_links():
    coaches_html = coaches_summary(to_pd=False)
    coach_urls = parse.links(coaches_html, prefix=sref.NFL_URL)
    return coach_urls


if __name__ == "__main__":
    ls = coach_links()
    print(ls)
