import os
import torch
import pytorch_lightning as pl
from trainer.supervised.trainer import SupervisedTrainerSkeleton

from models.efficientnet import EfficientNetModel
from models.project_layer import ProjectLayer

config_dir = os.path.join(os.getcwd(), "config", "efficientnet.yaml")

class EfficientNetTrainer(SupervisedTrainerSkeleton):
  def __init__(self, trainset = None, valset = None):
    super().__init__(yaml_file = config_dir, trainset = trainset, valset = valset)
    self.init_model()


  def init_model(self):
    self.encoder = EfficientNetModel(self.stream["backbone"], num_classes = self.stream["num_classes"])
    # Optimzer and scheduler
    self.configure_optimizers(self.stream)


  def get_grad_parameters(self):
    params = []
    for name, param in self.encoder.named_parameters():
      if param.requires_grad:
        params.append(param)

    return params



  def forward(self, images, labels):
    # Forward pass
    logits = self.encoder.forward(images)

    if self.stream["loss_func"] == "bceloss":
      logits = torch.sigmoid(logits)
      return self.loss_func(logits.to(torch.double), labels), logits
    else:
      return self.loss_func(logits, labels), logits


  def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
    self.optimizer.step()
    self.optimizer.zero_grad()

    self.scheduler.step()
