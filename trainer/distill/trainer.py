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

class DistillTrainerSkeleton(TrainerSkeleton):
  def __init__(self,
              yaml_file,
              institution = None,
              trainset = None,
              valset = None
              ):
    super().__init__(yaml_file, trainset, valset)
    self.institution = institution


  def training_step(self, train_batch, batch_idx):
    images, labels = train_batch
    loss, _  = self.forward(images, labels)
    logs = {"learning_rate": self.optimizer.param_groups[0]["lr"], "train_loss": loss.item()}
    return {"loss": loss, "log": logs}


  def validation_step(self, val_batch, val_idx):
    images, labels = val_batch
    val_loss, logits = self.forward(images, labels)

    preds = torch.max(F.softmax(logits, dim = -1), dim = -1)[1]
    logs = {"val_loss": val_loss,
            "y_pred": preds.cpu().numpy()}

    return logs