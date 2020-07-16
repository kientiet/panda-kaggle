from torch.optim.lr_scheduler import _LRScheduler


class ExponentialStep(_LRScheduler):
  def __init__(self, optimizer, decay_steps, decay_rate):
    super().__init__(optimizer, last_epoch = -1)

    self.decay_steps = decay_steps
    self.decay_rate = decay_rate
    self.num_epoch = 0


  def get_lr(self):
    if (self.last_epoch == 0) or (self.last_epoch % self.decay_steps != 0):
      return [group['lr'] for group in self.optimizer.param_groups]

    self.num_epoch += 1
    return [group["lr"] * pow(self.decay_rate, self.num_epoch)
            for group in self.optimizer.param_groups]
