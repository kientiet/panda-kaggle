import torch
import torch.nn as nn
import torchvision
from efficientnet_pytorch import EfficientNet

class EfficientNetModel(nn.Module):
  def __init__(self,
              model_name,
              **kwags
              ):

    super().__init__()
    self.encoder = EfficientNet.from_pretrained(model_name, num_classes = kwags["num_classes"])

  def forward(self, inputs):
    outputs = self.encoder.forward(inputs)
    return outputs