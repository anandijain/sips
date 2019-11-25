from sips.lines.bov import bov
from flask import Flask
import pandas as pd

df = bov.lines(['nba'])

app = Flask(__name__)


@app.route('/')
def hello_world():
    return df.to_html(header="true", table_id="table")


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
