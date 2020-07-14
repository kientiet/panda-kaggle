import torch
import torch.nn as nn
import torch.nn.functional as F


class OutliersLoss(nn.Module):
  def __init__(self, anneal = None, **kwags):
    super().__init__()
    self.anneal = anneal
    self.coeff = kwags["coeff"]
    self.temperature = kwags["temperature"]
    self.keep_classes = kwags["keep_classes"]

    # Default loss
    self.cross_entropy = nn.CrossEntropyLoss()
    self.kl_loss = nn.KLDivLoss(reduction = "sum")
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


  def forward(self, logits, hard_label, soft_logits, **kwags):
    hard_loss = self.cross_entropy(logits, hard_label)
    if kwags["running_mode"] == "training":
      student_prob, student_index = torch.sort(logits, dim = -1, descending = True)
      teacher_prob, teacher_index = torch.sort(soft_logits, dim = -1, descending = True)

      student_prob = student_prob[:, :self.keep_classes]
      teacher_prob = teacher_prob[:, :self.keep_classes]
      soft_loss = self.kl_loss(F.log_softmax(student_prob, dim = -1), F.softmax(teacher_prob, dim = -1))

      if not self.anneal or self.anneal is None:
        return hard_loss + self.coeff * soft_loss, hard_loss, soft_loss, self.coeff
      else:
        self.step()
        return self.coeff * hard_loss + (1 - self.coeff) * soft_loss, hard_loss, soft_loss, self.coeff
    else:
      return hard_loss