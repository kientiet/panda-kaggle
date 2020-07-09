import torch
import torch.nn as nn
import torchvision
from models.get_model import get_model
from efficientnet_pytorch import EfficientNet

from activation.mish import Mish

class EfficientNetModel(nn.Module):
  def __init__(self,
              model_name,
              **kwags
              ):

    super().__init__()
    self.encoder = EfficientNet(model_name)
    self.num_channel = list(self.encoder.children())[-1].in_features
    self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

  def forward(self, inputs):
    outputs = self.encoder(inputs)
    return outputs