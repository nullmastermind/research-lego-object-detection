import os

from pathlib import Path
from os.path import expanduser

dir_path = os.path.dirname(os.path.realpath(__file__))
home = expanduser("~")

drive = "train_data"


def init_drive():
    global drive

    dirs = [
        os.path.join(home, "Google Drive", "colab", "lego_yolov5"),
        os.path.join(Path(dir_path).parent, "drive", "MyDrive", "colab", "lego_yolov5"),
    ]

    for _dir in dirs:
        if Path(_dir).is_dir():
            drive = _dir
            return


init_drive()
