from sips.macros import sports_ref as sref
from sips.h import grab

def coaches_summary():
    coaches = grab.get_table(sref.fb_url + 'coaches/', ["coaches"])
    return coaches

def 