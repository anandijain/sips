from sips.macros import sports_ref as sref
from sips.h import grab
from sips.h import parse


def referee_links():
    """

    """
    links = []
    t = grab.get_table(sref.NBA_URL + "referees/", ["referees"], to_pd=False)
    ref_tags = t.find_all("th", {"data-stat": "referee"})
    for ref_tag in ref_tags:
        tag_link = ref_tag.find("a")
        if tag_link:
            links.append(tag_link["href"])
    return links


def ref_season_links():
    """

    """
    links = [
        sref.NBA_URL + "/referees/" + str(n) + "_register.html"
        for n in range(1989, 2020)
    ]

    refs = grab.tables_from_links(links, ["rs_raw"])
    return refs


if __name__ == "__main__":
    ls = referee_links()
    print(ls)

    ref_dfs = ref_season_links()
    print(ref_dfs)
