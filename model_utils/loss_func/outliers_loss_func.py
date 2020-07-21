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
    self.kl_loss = nn.KLDivLoss(reduction = "batchmean")

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


  def masked_softmax(self, logits, mask, dim = 1, epsilon = 1e-5):
      exps = torch.exp(logits)
      masked_exps = exps * mask.float()
      masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
      return masked_exps / masked_sums


  def compute_kl_loss(self, student_logits, teacher_logits):
    _, teacher_index = torch.sort(F.softmax(teacher_logits, dim = -1), dim = -1, descending = True)
    teacher_prob = torch.gather(teacher_logits, 1, teacher_index[:, :self.keep_classes])
    teacher_prob = F.softmax(teacher_prob, dim = -1)

    student_prob = torch.gather(student_logits, 1, teacher_index[:, :self.keep_classes])
    student_prob = F.log_softmax(student_prob, dim = -1)

    kl_loss = self.kl_loss(student_prob, teacher_prob)

    return kl_loss


  def forward(self, logits, hard_label, soft_logits, **kwags):
    hard_loss = self.cross_entropy(logits, hard_label)
    if kwags["running_mode"] == "training":
      kwags["optimizer"].zero_grad()

      soft_loss = self.compute_kl_loss(logits, soft_logits)
      # breakpoint()
      if not self.anneal or self.anneal is None:
        return hard_loss + self.coeff * soft_loss, hard_loss, soft_loss, self.coeff
      else:
        self.step()
        return self.coeff * hard_loss + (1 - self.coeff) * soft_loss, hard_loss, soft_loss, self.coeff
    else:
      return hard_loss