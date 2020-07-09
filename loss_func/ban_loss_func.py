import torch
import torch.nn as nn
import torch.nn.functional as F


class BanDistillLoss(nn.Module):
  def __init__(self, anneal = None, **kwags):
    super().__init__()
    self.anneal = anneal
    self.coeff = kwags["coeff"]
    self.temperature = kwags["temperature"]
    self.cross_entropy = nn.CrossEntropyLoss()
    if self.anneal:
      self.step_count(total_step = kwags["total_step"])


  def step_count(self, total_step):
    total_step = total_step
    self.inc_per_step = 1.0 / total_step
    self.coeff = -self.inc_per_step


  def step(self):
    self.coeff += self.inc_per_step


  def custom_cross_entropy(self, logits, labels):
    ce_loss = torch.sum(labels * (-F.log_softmax(logits, dim = -1)), dim = -1)
    return torch.mean(ce_loss)


  def forward(self, logits, hard_label, soft_label, **kwags):
    hard_loss = self.cross_entropy(logits, hard_label)
    if kwags["running_mode"] == "training":
      if kwags["label_type"] == "soft":
        soft_label = torch.pow(soft_label, 1 / self.temperature)
        soft_label = soft_label / torch.sum(soft_label, dim = -1, keepdim = True)
        soft_loss = self.custom_cross_entropy(logits, soft_label)
      else:
        soft_loss = self.cross_entropy(logits, soft_label)

      if self.anneal is None:
        return hard_loss + soft_loss * self.coeff, hard_loss, soft_loss
      else:
        self.step()
        return self.coeff * hard_loss + (1 - self.coeff) * soft_loss, hard_loss, soft_loss, self.coeff
    else:
      return hard_loss