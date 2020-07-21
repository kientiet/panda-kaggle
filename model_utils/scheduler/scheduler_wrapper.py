import yaml
import copy
import torch.optim.lr_scheduler as lr_scheduler

from model_utils.scheduler.lr_scheduler import exponential_decay, slant
from model_utils.scheduler.lr_scheduler.warmup_scheduler import GradualWarmupScheduler

scheduler_dict = {
  "1cycle": "model_utils/scheduler/config/1cycle.yaml",
  "cosine": "model_utils/scheduler/config/cosine.yaml",
  "slant": "model_utils/scheduler/config/slant.yaml",
  "step": "model_utils/scheduler/config/step.yaml",
  "exp": "model_utils/scheduler/config/exp.yaml",
  "warmup": "model_utils/scheduler/config/warmup.yaml"
}

class SchedulerWrapper(object):
  def __init__(self, scheduler_type, optimizer = None, **kwags):
    super().__init__()
    self.scheduler_type = scheduler_type
    self.total_epoch = kwags["total_epoch"]
    self.iteration_per_epoch = kwags["iteration_per_epoch"]
    self.warmup = kwags["warmup"]
    self.current_iteration = 0
    self.current_epoch = 0

    if optimizer is not None:
      kwags["max_lr_decay"] = False
      self.init_scheduler(optimizer, **kwags)


  def step(self):
    self.current_iteration += 1
    if self.is_warmup:
      self.scheduler.step(self.current_iteration)
    else:
      if self.scheduler_type in ["1cycle", "cosine", "slant"]:
        if self.current_iteration % (self.epochs * self.iteration_per_epoch + 1) == 0:
          self.init_scheduler(self.optimizer, max_lr_decay = True)
        else:
          self.scheduler.step()
      elif self.scheduler_type == "exp" or self.current_iteration == self.iteration_per_epoch:
        self.scheduler.step()

    if self.current_iteration % self.iteration_per_epoch == 0:
      self.current_epoch += 1



  def init_scheduler(self, optimizer, **kwags):
    self.is_warmup = False
    if "warmup" in self.scheduler_type:
      self.is_warmup = True
      scheduler_type = self.scheduler_type.split("_")
      self.scheduler_type = "_".join(scheduler_type[1:])

    yaml_config = scheduler_dict[self.scheduler_type]
    self.optimizer = optimizer

    with open(yaml_config, "r") as stream:
      stream = yaml.safe_load(stream)

      ## Set the attribute for wrapper
      if stream is not None:
        for key, value in stream.items():
          setattr(self, key, value)

      if self.scheduler_type == "1cycle":
        self.one_cycle_init(stream, **kwags)

      elif self.scheduler_type == "cosine":
        self.cosine(stream, **kwags)

      elif self.scheduler_type == "exp":
        self.exp_init(stream, **kwags)

      elif self.scheduler_type == "slant":
        self.slant(stream, **kwags)

    if self.is_warmup:
      yaml_config = scheduler_dict["warmup"]
      with open(yaml_config, "r") as stream:
        stream = yaml.safe_load(stream)

        ## Set the attribute for wrapper
        for key, value in stream.items():
          setattr(self, key, value)

        self.warmup_policy(stream, **kwags)


  def one_cycle_init(self, stream, **kwags):
    if kwags["max_lr_decay"]:
      stream["max_lr"] = stream["max_lr"] * stream["max_lr_decay_rate"] ** (self.current_epoch // stream["epochs"])

    if "max_lr_decay" in stream: stream.pop("max_lr_decay")
    stream.pop("max_lr_decay_rate")

    print("\n\n>> Running with 1cycle scheduler")
    stream["steps_per_epoch"] = self.iteration_per_epoch
    self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, **stream)


  def exp_init(self, stream, **kwags):
    print("\n\n>> Running with exp scheduler")
    stream["decay_steps"] = stream["decay_steps"] * self.iteration_per_epoch
    self.scheduler = exponential_decay.ExponentialStep(self.optimizer, **stream)


  def slant(self, stream, **kwags):
    print("\n\n>> Running with slant scheduler")
    self.epochs = self.total_epoch - self.warmup
    stream["steps_per_cycle"] = self.total_epoch * self.iteration_per_epoch
    self.scheduler = slant.STLR(self.optimizer, **stream)


  def cosine(self, stream, **kwags):
    print("\n\n>> Running with cosine scheduler")
    self.epochs = self.total_epoch
    stream = {"T_max": (self.total_epoch - self.warmup) * self.iteration_per_epoch}
    self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, **stream)


  def warmup_policy(self, stream, **kwags):
    self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier = stream["multiplier"], \
      total_epoch = self.warmup * self.iteration_per_epoch, after_scheduler = self.scheduler)