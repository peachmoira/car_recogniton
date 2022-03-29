import os
import sys
import pathlib
import torch
import numpy as np
from typing import List


from yolov5.models.experimental import attempt_load
#from yolov5.utils.datasets import letterbox
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device


class Detector(object):
    """

    """
    @classmethod
    def get_classname(cls: object) -> str:
        return cls.__name__

    def __init__(self) -> None:
        self.model = None
        self.device = "cpu"
        self.half = False

    def load_model(self, weights: str, device: str = '') -> None:
        device = select_device(device)
        model = attempt_load(weights, map_location=device)  # load FP32 model
        self.model = model
        self.device = device

    def load(self, path_to_model: str = "latest") -> None:
        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name("yolov5")
            path_to_model = model_info["path"]
        device = "cpu"
        if get_mode_torch() == "gpu":
            device = os.environ.get("CUDA_VISIBLE_DEVICES", '0')

        self.load_model(path_to_model, device)

    def detect_bbox(self,
                    img: np.ndarray,
                    img_size: int = 640,
                    stride: int = 32,
                    min_accuracy: float = 0.5) -> List:
        """
        TODO: input img in BGR format, not RGB; To Be Implemented in release 2.2
        """
        # normalize
        img_shape = img.shape
        img = letterbox(img, img_size, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
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
