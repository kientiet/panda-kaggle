import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from collections import Counter

class Evaluator:
  def __init__(self, valset, threshold = 0.5):
    self.valset = valset

  def evaluate_on_test_set(self, y_pred, y_pred_label = None):
    ## Split y_pred and y_true into dictionary
    y_true = self.valset.get_test_label()

    logs = {
      "kappa_score": cohen_kappa_score(y_true, y_pred, weights = "quadratic"),
      "accuracy_score": accuracy_score(y_true, y_pred)
    }

    return logs