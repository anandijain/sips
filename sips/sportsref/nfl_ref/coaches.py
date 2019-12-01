from sips.macros import sports_ref as sref
from sips.h import grab
from sips.h import parse


def coaches_summary(to_pd=True):
    coaches = grab.get_table(sref.nfl_url + "coaches/", ["coaches"], to_pd=to_pd)
    return coaches


def coach_links():
    coaches_html = coaches_summary(to_pd=False)
    coach_urls = parse.links(coaches_html, prefix=sref.nfl_url)
    return coach_urls


if __name__ == "__main__":
    ls = coach_links()
    print(ls)
