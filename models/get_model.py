import torch
import torchvision
from models.custom_resnet import custom_resnet34, custom_resnet18, custom_resnet50, \
                                custom_wide_resnet50_2, custom_wide_resnet50_4, custom_resnext50_32x4d

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

  "custom_resnet18": custom_resnet18,
  "custom_resnet34": custom_resnet34,
  "custom_resnet50": custom_resnet50,
  "custom_resnet50_32x4d": custom_resnext50_32x4d,
  "custom_wide_resnet50_2": custom_wide_resnet50_2,
  "custom_wide_resnet50_4": custom_wide_resnet50_4
}


def get_model(model_name, **kwags):
  assert model_name in model_dict
  return model_dict[model_name](**kwags)