import os
import base64

from flask import Flask, request
from flask_cors import CORS
from pathlib import Path

from yolo import init_yolo, detect

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello():
    return "Hello, World!"


@app.route("/label", methods=["POST"])
def label():
    req_data = request.get_json(force=True)
    print(req_data)
    save_dir = "train_data/projects/{}/{}".format(
        req_data["projectName"], req_data["dataType"]
    )
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    imgdata = base64.b64decode(req_data["imageBase64"])
    filename = next_path("{}/{}_%s.png".format(save_dir, req_data["label"]))

    with open(filename, "wb") as f:
        f.write(imgdata)
        print("saved: {}".format(filename))

    if req_data["autoMakeSense"]:
        result = detect(filename, req_data["objectConfidenceThreshold"])
        print(result)

    return "OK"


def next_path(path_pattern):
    """
    Finds the next free path in an sequentially named list of files

    e.g. path_pattern = 'file-%s.txt':

    file-1.txt
    file-2.txt
    file-3.txt

    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)

    while a + 1 < b:
        c = (a + b) // 2  # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return path_pattern % b


if __name__ == "__main__":
    init_yolo("train_data/best.pt")
    app.run(host="0.0.0.0")
