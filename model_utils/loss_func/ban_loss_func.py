import torch
import torch.nn as nn
import torch.nn.functional as F


class BanDistillLoss(nn.Module):
  def __init__(self, anneal = None, **kwags):
    super().__init__()
    self.cross_entropy = nn.CrossEntropyLoss()
    self.temperature = kwags["temperature"]


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
        soft_loss = self.cross_entropy(logits / self.temperature, soft_label)

      return soft_loss, hard_loss, soft_loss, 0.
    else:
      return hard_loss