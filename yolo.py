import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


model = None
device = None
half = None
imgsz = None
classify = False
modelc = None


def init_yolo(weights):
    global model
    global device
    global half
    global imgsz
    global classify
    global modelc

    # Initialize
    set_logging()
    device = select_device("")
    half = device.type != "cpu"

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(640, s=model.stride.max())  # check img_size

    if half:
        model.half()

    # Second-stage classifier
    if classify:
        modelc = load_classifier(name="resnet101", n=2)  # initialize
        modelc.load_state_dict(
            torch.load("weights/resnet101.pt", map_location=device)["model"]
        ).to(device).eval()


def detect(source, conf_thres=0.25, iou_thres=0.45, classes=None):
    global model
    global half
    global classify
    global modelc

    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != "cpu" else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred,
            conf_thres,
            iou_thres,
            classes=classes,
        )
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        result = []

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)
            s += "%gx%g " % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}s, "  # add to string
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (
                        (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                        .view(-1)
                        .tolist()
                    )  # normalized xywh
                    # line = (cls, *xywh, conf) if False else (cls, *xywh)  # label format
                    # print(("%g " * len(line)).rstrip() % line + "\n")
                    result.append((int(cls.item()), *xywh, conf.item()))

            # Print time (inference + NMS)
            print(f"{s}Done. ({t2 - t1:.3f}s)")
            return result
