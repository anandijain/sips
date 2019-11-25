import io
import random

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG

from matplotlib.figure import Figure

from sips.lines.bov import bov
from sips.h import helpers as h
from sips.h import serialize as s


import pandas as pd
import numpy as np

from flask import Flask, Response, request, render_template

# df = bov.lines(["nba"])
dfs = h.get_dfs()
sdfs = s.serialize_dfs(dfs, in_cols=['game_id', 'last_mod', 'a_ml', 'h_ml'], norm=False, to_numpy=False)
sdfs = [sdf for sdf in sdfs if not sdf.empty]
to_plot = random.choice(sdfs)
print(to_plot)

app = Flask(__name__)


def ml_ts(df):
    game_id = df.game_id.iloc[0]
    t = np.array(df.last_mod, dtype=np.float32)
    a_ml = np.array(df.a_ml, dtype=np.float32)
    h_ml = np.array(df.h_ml, dtype=np.float32)
    return game_id, t, a_ml, h_ml


@app.route("/")
def hello_world():
    return to_plot.to_html(header="true", table_id="table")


@app.route("/games")
def plot_game():
    game_id, t, a_ml, h_ml = ml_ts(to_plot)
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(t, a_ml)
    axis.plot(t, h_ml)
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)    
    return Response(output.getvalue(), mimetype="image/png")


if __name__ == "__main__":
    import webbrowser
    host = "0.0.0.0"
    app.run(host=host, debug=True)

    webbrowser.open("http://" + host + ":5000/")
