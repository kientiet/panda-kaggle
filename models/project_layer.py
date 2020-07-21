import torch
import torch.nn as nn
import numpy as np

from model_utils.activation.mish import Mish

class ProjectLayer(nn.Module):
  def __init__(self,
              num_channel,
              num_classes,
              is_pooling = True
              ):

    super().__init__()
    # Project head
    self.avg_pooling = nn.AdaptiveAvgPool2d(1)
    self.max_pooling = nn.AdaptiveMaxPool2d(1)
    self.head = nn.Sequential(nn.Flatten(),
                              nn.Linear(2 * num_channel, 512),
                              Mish(),
                              nn.BatchNorm1d(512),
                              nn.Dropout(0.5),
                              nn.Linear(512, num_classes)
                              )
    self.is_pooling = is_pooling
    self.init_weight()

  def init_weight(self):
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, np.sqrt(2. / (m.out_features * m.in_features)))


  def forward(self, inputs):
    if self.is_pooling:
      avg_pooling = self.avg_pooling(inputs)
      max_pooling = self.max_pooling(inputs)
      inputs = torch.cat((avg_pooling, max_pooling), dim = 1)

    outputs = self.head(inputs)
    return outputs