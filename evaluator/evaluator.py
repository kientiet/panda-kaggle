import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from collections import Counter

class TestSetEvaluator:
  def __init__(self, valset, threshold = 0.5):
    self.valset = valset
    self.threshold = threshold

  def eval_all(self, y_pred, y_true, y_pred_label = None):
    logs = self.eval_pictures_level(y_pred, y_true, y_pred_label)

    return logs

  def eval_pictures_level(self, y_pred, y_true, y_pred_label = None):
    kappa_score = cohen_kappa_score(y_true, y_pred, weights = "quadratic")

    # Get the label here
    logs = {
      "kappa_score": kappa_score,
      "accuracy_score": accuracy_score(y_true, y_pred)
    }

    return logs