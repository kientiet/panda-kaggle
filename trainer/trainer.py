import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np

from utils.preprocessing.split.supervised.train_test_split import load_dataframe
from utils.evaluator.evaluator import Evaluator

from model_utils.optimizer.radam import Over9000, LookaheadAdam
from model_utils.loss_func.get_loss_func import get_loss_func
from model_utils.scheduler.scheduler_wrapper import SchedulerWrapper

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
        self.trainset, self.valset = load_dataframe([stream["train_aug"], stream["test_aug"]], stream["data_dir"], \
          stream["loss_func"], stream["sample"])
      else:
        print(">> Assign exist data frame")
        self.trainset = trainset
        self.valset = valset

      self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size = stream["batch_size"], shuffle = True, num_workers = 4)
      self.valloader = torch.utils.data.DataLoader(self.valset, batch_size = stream["batch_size"], shuffle = False, num_workers = 4)

      self.temperature = stream["temperature"]
      self.stream = stream

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
    logs = {"learning_rate": self.optimizer.param_groups[-1]["lr"], "train_loss": loss.item()}
    return {"loss": loss, "log": logs}


  def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
    self.optimizer.step()
    self.optimizer.zero_grad()

    if hasattr(self, "project_layer"):
      weight = self.project_layer.head[1].weight.view(-1)
      self.logger.experiment.add_histogram("project_layer_1", weight, self.global_step)

      weight = self.project_layer.head[5].weight.view(-1)
      self.logger.experiment.add_histogram("project_layer_5", weight, self.global_step)

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

      # Scheduler
      self.scheduler_type = stream["scheduler_type"]
      self.cool_down = stream["cool_down"]
      self.warmup = stream["warmup"]
      self.total_epoch = stream["total_epoch"]

    self.reset_optimizer()
    self.reset_scheduler()

    return self.optimizer

  def get_grad_parameters(self):
    pass

  def reset_optimizer(self):
    # Initialize optimizer
    params = self.get_grad_parameters()

    if self.optimizer_type == "SGD":
      self.optimizer = optim.SGD(params,
                                lr = self.max_lr,
                                momentum = self.max_momentum,
                                weight_decay = self.weight_decay,
                                nesterov = self.nestrov)
    elif self.optimizer_type == "Adam":
      self.optimizer = optim.Adam(params, lr = self.max_lr, weight_decay = self.weight_decay)
    elif self.optimizer_type == "Over9000":
      self.optimizer = Over9000(params, lr = self.max_lr, weight_decay = self.weight_decay, betas=(0.9, 0.999))
    elif self.optimizer_type == "lookahead":
      self.optimizer = LookaheadAdam(params, alpha = 0.6, k = 6, lr = self.max_lr, weight_decay = self.weight_decay, betas=(0.8, 0.999))
    elif self.optimizer_type == "RMSprop":
      self.optimizer = optim.RMSprop(params, lr = self.max_lr, weight_decay = self.weight_decay, momentum = 0.9)


  def reset_scheduler(self):
    # Initialize scheduler
    self.scheduler = SchedulerWrapper(self.scheduler_type,
                                      self.optimizer,
                                      total_epoch = self.get_max_epoches(),
                                      iteration_per_epoch = len(self.trainloader),
                                      warmup = self.warmup)

  def freeze_model(self):
    print(">> Freeze all except batchnorm")
    self.freeze()
    for name, p in self.encoder.named_parameters():
      if "bn" in name:
        p.requires_grad = True

    for p in self.project_layer.parameters():
      p.requires_grad = True

    print(">> Reload the dataset and dataloader")
    self.trainset, self.valset = load_dataframe([self.stream["train_transformation"], self.stream["test_transformation"]], self.stream["fix_res"], \
      self.stream["loss_func"], self.stream["sample"])
    self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size = self.stream["batch_size"], shuffle = True, num_workers = 4)
    self.valloader = torch.utils.data.DataLoader(self.valset, batch_size = self.stream["batch_size"], shuffle = False, num_workers = 4)

    print(">> Config parameters")
    self.configure_optimizers()


  def validation_epoch_end(self, outputs):
    y_pred, y_true, data_provider = [], [], []
    for batch in outputs:
      y_pred = np.concatenate((y_pred, batch["y_pred"]))
      if "y_true" in batch:
        y_true = np.concatenate((y_true, batch["y_true"]))
        data_provider = np.concatenate((data_provider, batch["data_provider"]))

    tensorboard_logs, total_fig, fig_karolinska, fig_radboud = self.evaluator.evaluate_on_test_set(y_pred = y_pred, y_true = y_true, \
      data_provider = data_provider)
    if total_fig is not None: self.logger.experiment.add_figure("confusion_matrix", total_fig, self.at_epoch)
    if fig_karolinska is not None: self.logger.experiment.add_figure("karolinska_confusion_matrix", fig_karolinska, self.at_epoch)
    if fig_radboud is not None: self.logger.experiment.add_figure("radboud_confusion_matrix", fig_radboud, self.at_epoch)

    # Get the validation loss
    total_loss = torch.stack([batch["val_loss"] for batch in outputs]).mean()
    tensorboard_logs["val_loss_epoch"] = total_loss.item()

    logs = {"val_loss": total_loss, "log": tensorboard_logs}
    return logs


  def on_epoch_end(self):
    self.at_epoch += 1

  def get_max_epoches(self):
    return self.total_epoch + self.cool_down + self.warmup


  def get_total_step(self):
    return self.get_max_epoches() * len(self.trainloader)


  def train_dataloader(self):
    print(">> Return trainloader")
    return self.trainloader


  def val_dataloader(self):
    print(">> Return valloader")
    # Import the evaluator
    self.evaluator = Evaluator(self.valset)

    return self.valloader