import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from collections import Counter


class Evaluator:
  def __init__(self, valset, threshold = 0.5):
    self.valset = valset

  def evaluate_on_test_set(self, y_pred, y_pred_label = None):
    ## Split y_pred and y_true into dictionary
    y_true, table = self.valset.get_test_label()
    total_fig = self.plot_confusion_matrix(y_true, y_pred)

    raboud_index = table["data_provider"] == "radboud"
    fig_radboud = self.plot_confusion_matrix(y_true[raboud_index], y_pred[raboud_index])

    karolinska_index = table["data_provider"] == "karolinska"
    fig_karolinska = self.plot_confusion_matrix(y_true[karolinska_index], y_pred[karolinska_index])

    logs = {
      "kappa_score/kappa_score": cohen_kappa_score(y_true, y_pred, weights = "quadratic"),
      "kappa_score/radboud": cohen_kappa_score(y_true[raboud_index], y_pred[raboud_index], weights = "quadratic"),
      "kappa_score/karolinska": cohen_kappa_score(y_true[karolinska_index], y_pred[karolinska_index], weights = "quadratic"),

      "accuracy_score/total": accuracy_score(y_true, y_pred),
      "accuracy_score/radboud": accuracy_score(y_true[raboud_index], y_pred[raboud_index]),
      "accuracy_score/karolinska": accuracy_score(y_true[karolinska_index], y_pred[karolinska_index]),
    }

    return logs, total_fig, fig_karolinska, fig_radboud

  def plot_confusion_matrix(self, y_true, y_pred, normalized = False):
    cm = confusion_matrix(y_true, y_pred)

    if normalized:
      cm = cm.astype('float')*10 / cm.sum(axis = 1)[:, np.newaxis]
      cm = np.nan_to_num(cm, copy = True)
      cm = cm.astype('int')

    np.set_printoptions(precision = 2)

    fig = plt.figure(figsize = (7, 7), dpi = 130, facecolor = "w", edgecolor = "k")
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap = "Oranges")

    classes = np.arange(np.max(y_true) + 1)
    tick_marks = np.arange(len(classes))

    ax.set_xlabel("Predicted", fontsize = 7)
    ax.set_xticks(tick_marks)

    c = ax.set_xticklabels(classes, fontsize = 4, rotation = -90, ha = "center")
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)

    return fig