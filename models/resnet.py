import torch
import torch.nn as nn
import torchvision

from model_utils.activation.mish import Mish


def load_resnet18(pretrained):
  if pretrained:
    return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet18_swsl')

  return torchvision.models.resnet18(pretrained = False)


def load_resnet50(pretrained):
  if pretrained:
    return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')

  return torchvision.models.resnet50(pretrained = False)


def load_resnext50_32x4d(pretrained):
  if pretrained:
    return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_swsl')

  return torchvision.models.resnext50_32x4d(pretrained = False)


model_dict = {
  "resnet18": load_resnet18,
  "resnet50": load_resnet50,
  "resnet50_32x4d": load_resnext50_32x4d,
}


def get_model(model_name, **kwags):
  assert model_name in model_dict
  return model_dict[model_name](**kwags)


class ResNetModel(nn.Module):
  def __init__(self,
              model_name,
              **kwags
              ):

    super().__init__()
    self.encoder = get_model(model_name, **kwags)
    self.num_channel = list(self.encoder.children())[-1].in_features
    self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

  def forward(self, inputs, index = 0):
    outputs = self.encoder((inputs, index))
    return outputs