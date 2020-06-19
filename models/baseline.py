import torch
import torch.nn as nn
import torchvision

class BaseLineModel(nn.Module):
  def __init__(self,
              num_classes = 5,
              dropout = 0.1
              ):

    super().__init__()
    self.model = torchvision.models.resnet18(pretrained = True)
    self.fc = nn.Linear(1000, num_classes)

    self.dropout = None
    if dropout > 0:
      self.dropout = nn.Dropout(dropout)

    self.activation = nn.ReLU(inplace = True)

  def forward(self, inputs):
    features = self.model(inputs)

    if self.dropout is not None:
      features = self.dropout(features)

    outputs = self.fc(features)
    outputs = self.activation(outputs)
    return outputs