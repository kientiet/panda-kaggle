import torch
import torch.nn as nn
import torchvision
from models.get_model import get_model

from model_utils.activation.mish import Mish

class ResNetModel(nn.Module):
  def __init__(self,
              model_name,
              **kwags
              ):

    super().__init__()
    self.encoder = get_model(model_name, **kwags)
    self.num_channel = list(self.encoder.children())[-1].in_features
    self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

  def forward(self, inputs):
    outputs = self.encoder(inputs)
    return outputs