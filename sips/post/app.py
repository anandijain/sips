"""

"""
from flask import Flask, jsonify, request
import io
import random
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG

from matplotlib.figure import Figure

import tensorflow as tf

import pandas as pd
import numpy as np

from sips.lines.bov import bov
from sips.h import helpers as h
from sips.h import serialize as s
from sips.h import calc

from flask import Flask, Response, request, render_template, jsonify


dfs = h.get_dfs()
sdfs = s.serialize_dfs(
    dfs, norm=False, to_numpy=False, dont_hot=True, output_type="dict"
)
print(sdfs.keys())
to_plot = random.choice(list(sdfs.items()))[1]
# print(to_plot)

app = Flask(__name__)


imagenet_class_index = json.load(
    open("/home/sippycups/absa/sips/sips/post/imagenet_class_index.json")
)

model = models.densenet121(pretrained=True)
model.eval()


def ml_ts(df):
    """

    """
    t = np.array(df.last_mod, dtype=np.float32)
    a_ml = np.array(df.a_ml, dtype=np.float32)
    h_ml = np.array(df.h_ml, dtype=np.float32)
    return t, a_ml, h_ml


@app.route("/")
def hello_world():
    """

    """
    return to_plot.to_html(header="true", table_id="table")


@app.route("/lines")
def lines():
    """

    """
    df = bov.lines(["nba", "nfl"])
    # sdf = s.serialize_df(df, label_cols=["a_ml", "h_ml"], to_numpy=False)
    # X = sdf[0].to_html(header="true", table_id="lines")
    # Y = sdf[1].to_html(header="true", table_id="lines")
    return df.to_html(header="true", table_id="lines")


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


@app.route("/predict", methods=["POST"])
def predict():
    """

    """

    if request.method == "POST":
        # we will get the file from the request
        file = request.files["file"]
        # convert that to bytes
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({"class_id": class_id, "class_name": class_name})


def transform_image(image_bytes):
    my_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


@app.route("/<int:game_id>")
def plot_game(game_id):
    """

    """
    fig = Figure(figsize=(7, 10))

    t, a_ml, h_ml = ml_ts(sdfs[game_id])

    a_deci = list(map(calc.eq, a_ml))
    h_deci = list(map(calc.eq, h_ml))
    print(f"a_deci: {a_deci}")
    print(f"h_deci: {h_deci}")
    axis = fig.add_subplot(1, 1, 1, title=str(game_id))
    axis.plot(t, a_ml, label="a_ml")
    axis.plot(t, h_ml, label="h_ml")
    axis.plot(t, a_deci, label="a_deci")
    axis.plot(t, h_deci, label="h_deci")
    # axis.set_yscale('log')
    fig.legend(loc="best")
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)

    return Response(output.getvalue(), mimetype="image/png")


app = Flask(__name__)
imagenet_class_index = json.load(
    open("/home/sippycups/absa/sips/sips/post/imagenet_class_index.json")
)
model = models.densenet121(pretrained=True)
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({"class_id": class_id, "class_name": class_name})


if __name__ == "__main__":
    import webbrowser

    host = "0.0.0.0"
    app.run(host=host, debug=True)
    ids = list(sdfs.keys())
    plot_game("6006183")

    webbrowser.open("http://" + host + ":5000/")
