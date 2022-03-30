import torch
from torch.nn import functional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from collections import Counter


from ocrr.numberplate_ocr_data_module import OcrNetDataModule
from ocrr.ocr_model import NPOcrNet, weights_init
from ocrr.data_loaders import (normalize, aug_seed)
from ocrr.ocr_tools import (StrLabelConverter,
                                         decode_prediction,
                                         decode_batch)


mode_torch = 'gpu'
ocr_path = 'models/anpr_ocr_kz_2021_09_01_pytorch_lightning.ckpt'

class OCR(object):
    def __init__(self) -> None:
        # model
        self.dm = None
        self.model = None
        self.trainer = None
        self.letters = []
        self.max_text_len = 0

        # Input parameters
        self.max_plate_length = 0
        self.height = 50
        self.width = 200
        self.color_channels = 3
        self.label_length = 13

        # Train hyperparameters
        self.batch_size = 32
        self.epochs = 1
        self.gpus = 1

        self.label_converter = None


    def load_model(self, path_to_model):
        if mode_torch == "gpu":
            self.model = NPOcrNet.load_from_checkpoint(path_to_model,
                                                       map_location=torch.device('cuda'),
                                                       letters=self.letters,
                                                       letters_max=len(self.letters) + 1,
                                                       img_h=self.height,
                                                       img_w=self.width,
                                                       label_converter=self.label_converter,
                                                       max_plate_length=self.max_plate_length)
        else:
            self.model = NPOcrNet.load_from_checkpoint(path_to_model,
                                                       map_location=torch.device('cpu'),
                                                       letters=self.letters,
                                                       letters_max=len(self.letters) + 1,
                                                       img_h=self.height,
                                                       img_w=self.width,
                                                       label_converter=self.label_converter,
                                                       max_plate_length=self.max_plate_length)
        self.model.eval()
        return self.model


o = OCR()
print(o.load_model(ocr_path))


