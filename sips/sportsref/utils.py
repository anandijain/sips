



def url_to_id(url: str) -> str:
    return url.split("/")[-1].split(".")[0]
