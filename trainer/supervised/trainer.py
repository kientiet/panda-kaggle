import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np

from trainer.trainer import TrainerSkeleton
from model_utils.loss_func.get_loss_func import get_loss_func
from utils.evaluator.evaluator import Evaluator

class SupervisedTrainerSkeleton(TrainerSkeleton):
  def __init__(self,
              yaml_file,
              trainset = None,
              valset = None
              ):
    super().__init__(yaml_file, trainset, valset)

  def validation_step(self, val_batch, val_idx):
    self.eval()
    with torch.no_grad():
      images, labels = val_batch
      val_loss, logits = self.forward(images, labels)

    if self.stream["loss_func"] == "bceloss":
      preds = logits.sum(1).detach().round()
    else:
      preds = torch.max(F.softmax(logits, dim = -1), dim = -1)[1]

    logs = {"val_loss": val_loss,
            "y_pred": preds.cpu().numpy()}

    return logs