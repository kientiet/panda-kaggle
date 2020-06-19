import torch
import torch.nn as nn

loss_dictionary = {
  "crossentropy": nn.CrossEntropyLoss()
}

def get_loss_func(loss_name):
  assert loss_name in loss_dictionary
  return loss_dictionary[loss_name]