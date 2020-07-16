import torch
import torch.nn as nn

from model_utils.loss_func.outliers_loss_func import OutliersLoss
from model_utils.loss_func.bam_loss_func import BamDistillLoss


loss_dictionary = {
  "crossentropy": nn.CrossEntropyLoss(),
  "bceloss": nn.BCELoss(),
  "outliers_loss_func": OutliersLoss,
  "bam_loss_func": BamDistillLoss
}

def get_loss_func(loss_name):
  assert loss_name in loss_dictionary
  return loss_dictionary[loss_name]