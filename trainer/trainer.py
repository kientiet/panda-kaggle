import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np

from preprocessing.split.supervised_train_test_split import load_dataframe
from loss_func.get_loss_func import get_loss_func
from evaluator.evaluator import Evaluator

class TrainerSkeleton(pl.LightningModule):
  def __init__(self,
              yaml_file,
              trainset = None,
              valset = None
              ):
    super().__init__()

    # Load the necessary hyperparameters
    with open(yaml_file, "r") as stream:
      stream = yaml.safe_load(stream)
      self.model_name = stream["model_name"]

      # Load data
      if trainset is None:
        print(">> Load from data frame")
        self.trainset, self.valset = load_dataframe([stream["train_transformation"], stream["test_transformation"]], stream["data_dir"], stream["loss_func"])
      else:
        print(">> Assign exist data frame")
        self.trainset = trainset
        self.valset = valset

      self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size = stream["batch_size"], shuffle = True, num_workers = 4)
      self.valloader = torch.utils.data.DataLoader(self.valset, batch_size = stream["batch_size"], shuffle = False, num_workers = 4)

      self.stream = stream

    # Import the evaluator
    self.evaluator = Evaluator(self.valset)

    # Loss function
    self.loss_func = get_loss_func(stream["loss_func"])

    self.at_epoch = 0


  def init_model(self):
    pass


  def forward(self, inputs, labels):
    pass


  def training_step(self, train_batch, batch_idx):
    images, labels = train_batch
    loss, _  = self.forward(images, labels)
    logs = {"learning_rate": self.optimizer.param_groups[0]["lr"], "train_loss": loss.item()}
    return {"loss": loss, "log": logs}


  def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
    self.optimizer.step()
    self.optimizer.zero_grad()

    if self.scheduler_type in ["1cycle", "cosine"]:
      self.scheduler.step()



  def configure_optimizers(self, stream = None):
    if stream:
      # Optimizer config
      self.optimizer_type = stream["optimizer_type"]
      self.max_lr = stream["max_lr"]
      self.base_momentum = stream["base_momentum"]
      self.max_momentum = stream["max_momentum"]
      self.nestrov = stream["nestrov"]
      self.weight_decay = stream["weight_decay"]

      # Scheduler config
      self.scheduler_type = stream["scheduler_type"]
      self.num_cycle = stream["num_cycle"]
      self.epoch_per_cycle = stream["epoch_per_cycle"]
      self.warmup = stream["warmup"]
      self.cool_down = stream["cool_down"]

    self.reset_optimizer()
    self.reset_scheduler()

    return self.optimizer


  def reset_optimizer(self):
    # Initialize optimizer
    if self.optimizer_type == "SGD":
      self.optimizer = optim.SGD(self.encoder.parameters(),
                                lr = self.max_lr,
                                momentum = self.max_momentum,
                                weight_decay = self.weight_decay,
                                nesterov = self.nestrov)
    else:
      self.optimizer = optim.Adam(self.encoder.parameters(), lr = self.max_lr, weight_decay = self.weight_decay)


  def reset_scheduler(self):
    # Initialize scheduler
    if self.scheduler_type == "1cycle":
      self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                    max_lr = self.max_lr,
                                                    epochs = self.epoch_per_cycle,
                                                    steps_per_epoch = len(self.trainloader),
                                                    base_momentum = self.base_momentum,
                                                    max_momentum = self.max_momentum,
                                                    div_factor = 25,
                                                    final_div_factor = 100,
                                                    pct_start = 0.0
                                                    )
    elif self.scheduler_type == "cosine":
      self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                            T_max = self.num_cycle * len(self.trainloader) * 16.0 / 7.0)
    elif self.scheduler_type == "linear":
      self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = self.decay_epoch, gamma = 0.1)


  def validation_epoch_end(self, outputs):
    if self.scheduler_type == "linear":
      self.scheduler.step()

    y_pred = []
    for batch in outputs:
      y_pred = np.concatenate((y_pred, batch["y_pred"]))

    tensorboard_logs = self.evaluator.evaluate_on_test_set(y_pred = y_pred)

    # Get the validation loss
    total_loss = torch.stack([batch["val_loss"] for batch in outputs]).mean()
    tensorboard_logs["val_loss_epoch"] = total_loss.item()

    logs = {"val_loss": total_loss, "log": tensorboard_logs}
    return logs


  def on_epoch_end(self):
    self.at_epoch += 1
    if self.at_epoch % self.epoch_per_cycle == 0:
      self.reset_scheduler()


  def get_max_epoches(self):
    if self.num_cycle > 0:
      return self.num_cycle * self.epoch_per_cycle + self.cool_down + self.warmup
    else:
      return self.total_iteration + self.cool_down + self.warmup


  def get_total_step(self):
    return (self.num_cycle * self.epoch_per_cycle + self.cool_down + self.warmup) * len(self.trainloader)


  def train_dataloader(self):
    print(">> Return trainloader")
    return self.trainloader


  def val_dataloader(self):
    print(">> Return valloader")
    return self.valloader