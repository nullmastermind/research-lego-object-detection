from pathlib import Path

drive = "../drive/MyDrive/colab/lego_yolov5"

if not Path(drive).is_dir():
    drive = "train_data"
