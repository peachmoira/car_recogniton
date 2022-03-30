import numpy as np
import cv2
import torch
from typing import List

from yolov5.utils.datasets import letterbox
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device


class Detector(object):
    def __init__(self) -> None:
        self.model = None
        self.device = "cpu"

    def load_model(self, weights: str, device: str = '') -> None:
        device = select_device(device)
        model = attempt_load(weights, map_location=device)  # load FP32 model

        self.model = model
        self.device = device

    def detect_bbox(self,
                    img: np.ndarray,
                    img_size: int = 640,
                    stride: int = 32,
                    min_accuracy: float = 0.5) -> List:
        # normalize
        img_shape = img.shape
        img = letterbox(img, img_size, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(np.flip(img,axis=0).copy()).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred)
        res = []
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_shape).round()
                res.append(det.cpu().detach().numpy())
        if len(res):
            return [[x1, y1, x2, y2, acc, b] for x1, y1, x2, y2, acc, b in res[0] if acc > min_accuracy]
        else:
            return []


