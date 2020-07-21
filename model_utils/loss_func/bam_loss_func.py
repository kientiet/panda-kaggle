import torch
import torch.nn as nn
import torch.nn.functional as F

class BamDistillLoss(nn.Module):
  def __init__(self, anneal = None, **kwags):
    super().__init__()
    self.anneal = anneal
    self.coeff = kwags["coeff"]
    self.temperature = kwags["temperature"]
    self.label_type = kwags["label_type"]

    # Default loss
    self.cross_entropy = nn.CrossEntropyLoss(reduction = "none")

    if self.anneal:
      self.step_count(total_step = kwags["total_step"])
    else:
      self.inc_per_step = 0


  def step_count(self, total_step):
    total_step = total_step
    self.inc_per_step = 1.0 / total_step
    self.coeff = -self.inc_per_step


  def step(self):
    self.coeff += self.inc_per_step

  def custom_cross_entropy(self, logits, soft_logits):
    ce_loss = torch.sum(soft_logits * (-F.log_softmax(logits, dim = -1)), dim = -1)
    return ce_loss


  def forward(self, logits, hard_label, **kwags):
    hard_loss = self.cross_entropy(logits, hard_label)
    if kwags["running_mode"] == "training":
      soft_labels = kwags["soft_labels"]
      task_weights = kwags["task_weights"]
      if self.label_type == "soft":
        soft_loss = self.custom_cross_entropy(logits, soft_labels)
      else:
        soft_labels = torch.max(soft_labels, dim = -1)[1]
        soft_loss = self.cross_entropy(logits, soft_labels)

      if not self.anneal or self.anneal is None:
        total_loss = torch.mean((hard_loss + self.coeff * soft_loss) * task_weights)
      else:
        self.step()
        total_loss = torch.mean((self.coeff * hard_loss + (1 - self.coeff) * soft_loss) * task_weights)

      return total_loss, torch.mean(hard_loss), torch.mean(soft_loss), self.coeff
    else:
      return torch.mean(hard_loss)