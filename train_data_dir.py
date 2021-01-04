from pathlib import Path
from os.path import expanduser

home = expanduser("~")

drive = "train_data"


def init_drive():
    global drive

    dirs = [
        "{}/Google Drive/colab/lego_yolov5".format(home),
        "../drive/MyDrive/colab/lego_yolov5",
    ]

    for _dir in dirs:
        if Path(_dir).is_dir():
            drive = _dir
            return


init_drive()
