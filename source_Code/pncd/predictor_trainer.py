import torch

from source_Code.pncd.adaptive_predictor import AdaptivePredictor
from source_Code.pncd.config import *

predictor = AdaptivePredictor()

optimizer = torch.optim.Adam(

    predictor.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)