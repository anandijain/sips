def divy_games(events):
    """

    we want to divy up the games by locations closest to our server.
    prereq: dataframe for arena latitude and longitudes,
        as well as the events

    1. determine location of each game based on home away (in terms of arena)
    2. lookup arena to find latitude and longitude forall games in dataframe
    3. compute distance to game to each of the 3 servers (LA, Chi, NY)
    4. log the games for each location
    5. 

    need a way to prevent requests to all_events 

    # testing
    1. improve logging to capture the request time of every request as option
    2. break down timing to see if the bulk of time is in latency via reqs
        or in local allocs / fileio etc
    3. if the bulk of time is spent on requests, it might be faster to req 
        all events every time (even just as a new-game check)


    """
