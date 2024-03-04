import torch
import torch.optim as optimizer
from model import Yolov1
from loss import Loss

DEVICE = 'cpu'

def predict():
    model = Yolov1(num_spilt=7, num_boxes=2, num_classes=20).to_device(DEVICE)