# Specify device
import os

# Specify device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import all necessary libraries.
import sys
import cv2

# NomeroffNet path
NOMEROFF_NET_DIR = os.path.abspath('../')

sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet.YoloV5Detector import Detector
detector = Detector()
detector.load()

from NomeroffNet.BBoxNpPoints import NpPointsCraft, getCvZoneRGB, convertCvZonesRGBtoBGR, reshapePoints
npPointsCraft = NpPointsCraft()
npPointsCraft.load()

from NomeroffNet.OptionsDetector import OptionsDetector

from NomeroffNet.TextDetectors.eu import eu
from NomeroffNet import textPostprocessing

# load models
optionsDetector = OptionsDetector()
optionsDetector.load("latest")

textDetector = eu()
textDetector.load("latest")

# Detect numberplate
img_path = 'examples/images/example2.jpeg'
img = cv2.imread(img_path)
img = img[..., ::-1]

targetBoxes = detector.detect_bbox(img)
all_points = npPointsCraft.detect(img, targetBoxes,[5,2,0])

# cut zones
zones = convertCvZonesRGBtoBGR([getCvZoneRGB(img, reshapePoints(rect, 1)) for rect in all_points])

# predict zones attributes
regionIds, countLines = optionsDetector.predict(zones)
regionNames = optionsDetector.getRegionLabels(regionIds)

# find text with postprocessing by standart
textArr = textDetector.predict(zones)
textArr = textPostprocessing(textArr, regionNames)
print(textArr)
# ['JJF509', 'RP70012']