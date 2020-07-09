import torch
import torch.nn as nn

from loss_func.ban_loss_func import BanDistillLoss

loss_dictionary = {
  "crossentropy": nn.CrossEntropyLoss(),
  "bceloss": nn.BCELoss(),
  "ban_loss_func": BanDistillLoss
}

def get_loss_func(loss_name):
  assert loss_name in loss_dictionary
  return loss_dictionary[loss_name]