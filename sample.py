#models
yolo_path = 'models/yolov5s-2021-12-14.pt'
trained_model = 'models/craft_mlt_25k_2020-02-16.pth'
refiner_model = 'models/craft_refiner_CTW1500_2020-02-16.pth'
ocr_path = 'anpr_ocr_kz_2021_09_01_pytorch_lightning.ckpt'

import os

# Specify device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import all necessary libraries.
import sys
import cv2


# Import license plate recognition tools.
from yolo.yolovDetect import Detector

detector = Detector()
detector.load_model(yolo_path)

from ocrr.kz import kz
# from NomeroffNet import textPostprocessing

textDetector = kz()
textDetector.load_model(ocr_path)

# Detect numberplate
img_path = 'mashina.jpg'
img = cv2.imread(img_path)
img = img[..., ::-1]

targetBoxes = detector.detect_bbox(img)
print(targetBoxes)

zones = []
regionNames = []
for targetBox in targetBoxes:
    x = int(min(targetBox[0], targetBox[2]))
    w = int(abs(targetBox[2] - targetBox[0]))
    y = int(min(targetBox[1], targetBox[3]))
    h = int(abs(targetBox[3] - targetBox[1]))

    image_part = img[y:y + h, x:x + w]
    zones.append(image_part)
    regionNames.append('kz')

# find text with postprocessing by standart
textArr = textDetector.predict(zones)
print(textArr)
# textArr = textPostprocessing(textArr, regionNames)
# print(textArr)
# ['RP70012', 'JJF509']