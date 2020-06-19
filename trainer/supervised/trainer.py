import yaml
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from loss_func.get_loss_func import get_loss_func

class TrainerSkeleton(pl.LightningModule):
  def __init__(self,
              trainloader,
              valloader,
              valset,
              yaml_file
              ):
    super(TrainerSkeleton, self).__init__()

    # Load data
    self.trainloader = trainloader
    self.valloader = valloader
    self.valset = valset

    # Load the necessary hyperparameters
    with open(yaml_file, "r") as stream:
      stream = yaml.safe_load(stream)

      self.running_scheduler = stream["running_scheduler"]
      self.base_lr = stream["base_lr"]
      self.max_lr = stream["max_lr"]
      self.num_cycle = stream["num_cycle"]
      self.epoch_per_cycle = stream["epoch_per_cycle"]
      self.cool_down = stream["cool_down"]

    self.current_step = 0
    self.current_epoch = 0

  def forward(self, *args, **kwargs):
    pass

  def training_step(self, train_batch, batch_idx):
    self.current_step += 1
    images, labels = train_batch
    logits = self.forward(images)
    loss = self.loss_func(logits, labels)
    logs = {"learning_rate": self.optimizer.param_groups[0]["lr"], "train_loss": loss.item()}
    return {"loss": loss, "log": logs}


  def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
    self.optimizer.zero_grad()
    self.optimizer.step()

    if self.running_scheduler and self.current_epoch < self.get_total_cycle():
      self.scheduler.step()


  def validation_step(self, val_batch, val_idx):
    images, labels = val_batch
    logits = self.forward(images)
    val_loss = self.loss_func(logits, labels)

    prob = F.softmax(logits, dim = -1)
    _, y_pred = torch.max(prob, dim = -1)

    logs = {"val_loss": val_loss,
            "y_pred": y_pred.cpu().numpy(),
            "y_true": labels.cpu().numpy()
            }

    return logs


  def validation_epoch_end(self, outputs):
    y_pred, y_true = []
    for batch in outputs:
      y_pred = np.concatenate((y_pred, batch["y_pred"]))
      y_true = np.concatenate((y_true, batch["y_true"]))

    tensorboard_logs = self.evaluator.evaluate_on_test_set(y_pred = y_pred, y_true = y_true)

    # Get the validation loss
    total_loss = torch.stack([batch["val_loss"] for batch in outputs]).mean()
    tensorboard_logs["val_loss_epoch"] = total_loss

    logs = {"val_loss": total_loss, "log": tensorboard_logs}
    return logs


  def train_loader(self):
    return self.trainloader

  def val_dataloader(self):
    return self.valloader