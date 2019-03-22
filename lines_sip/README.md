lines_sip is a tracker for bovada, primarily basketball

Open up a terminal/shell and navigate to where this project resides and execute the following for quick-start:
    
    python3
    import tests as t
    x = t.Test('0000', 1, 1, 1)  # these inputs explained below
    x.run()

The arguments that the Test function takes in are:
* 0: file_name (needs to be a csv)
* 1: game/sport type (see list of sport codes below)
* 2: if 1, write the column headers
* 3: if 1, run the program after initialization

sport codes:


    0 - All basketball
    1 - NBA
    2 - College Basketball
    3 - Hockey
    4 - tennis
    5 - Esports
    6 - Football
    7 - Volleyball


READ ABOVE FOR QUICKSTART

What is going on when the quickstart is run:

* First you start python with the python3 command.
* Then you import a python file called 'tests' and refer to it as 't'.
* Then you create an instantiation of the Test class in x.
* Test relies on the main python file that contains the scraper, sippy_lines.py.
