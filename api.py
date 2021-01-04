import os
import base64
import json

from flask import Flask, request
from flask_cors import CORS
from pathlib import Path

from yolo import init_yolo, detect
from labels import labels, save_label
from train_data_dir import drive

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello():
    return "Hello, World!"


@app.route("/label", methods=["POST"])
def label():
    req_data = request.get_json(force=True)
    print(req_data)
    root_save_dir = "{}/projects/{}/{}".format(
        drive, req_data["projectName"], req_data["dataType"]
    )
    save_dir = "{}/images".format(root_save_dir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    imgdata = base64.b64decode(req_data["imageBase64"])
    filename = next_path("{}/{}_%s.png".format(save_dir, req_data["label"]))

    with open(filename, "wb") as f:
        f.write(imgdata)
        f.close()
        print("saved: {}".format(filename))

    save_label(req_data["label"], root_save_dir)

    result = []
    if req_data["autoMakeSense"]:
        result = detect(filename, req_data["objectConfidenceThreshold"])
        if len(result) > 0:
            lines = ""
            for item in result:
                for v in item:
                    lines += str(v) + " "
                lines = lines.strip() + "\n"

            with open(filename + ".pre", "w") as f:
                f.write(lines.strip())
                f.close()

    return json.dumps([labels, result])


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
    print(drive)
    print(labels)
    init_yolo("{}/best.pt".format(drive))
    app.run(host="0.0.0.0")
