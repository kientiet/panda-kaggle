import torch
import torch.nn as nn
import torchvision
from models.custom_resnet import get_custom_model

from model_utils.activation.mish import Mish

class CustomResNetModel(nn.Module):
  def __init__(self,
              model_name,
              **kwags
              ):

    super().__init__()
    self.encoder = get_custom_model(model_name, **kwags)
    self.num_channel = self.encoder.num_channel

  def forward(self, inputs, index = 0):
    outputs = self.encoder(inputs, index)
    return outputs