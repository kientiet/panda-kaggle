from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.preprocessing.task_weight.config import Config
# from config import Config

def _multiples_and_weights(config):
  '''
    TODO: Generalize this for randomness
  '''
  dataset_sizes = {
    "karolinska": 11250,
    "radboud": 5538
  }

  def map_values(f, d):
    return {k: f(v) for k, v in d.items()}

  def map_kv(f, d):
    return {k: f(k, v) for k, v in d.items()}

  def normalize(d):
    total = float(sum(d.values()))
    return map_values(lambda v: v / total, d)

  dataset_weights = map_values(lambda s: s ** config.task_weight_exponent,
                               dataset_sizes)
  dataset_weights = normalize(dataset_weights)
  correction = dataset_sizes["karolinska"] / dataset_weights["karolinska"]
  dataset_tgts = map_values(lambda v: v * correction, dataset_weights)
  dataset_multiples = map_kv(
      lambda task, tgt: round((tgt + 0.01) / dataset_sizes[task]), dataset_tgts)
  new_dataset_sizes = map_kv(
      lambda task, multiple: dataset_sizes[task] * multiple, dataset_multiples)
  weights_after_multiples = map_values(
      lambda v: v * len(dataset_sizes),
      normalize({task: dataset_weights[task] / new_dataset_sizes[task]
                 for task in new_dataset_sizes}))

  print(weights_after_multiples)
  return dataset_multiples, weights_after_multiples


def get_task_multiple(task, split):
  if split != "train":
    return 1

  if task.dataset_multiples:
    multiples, _ = _multiples_and_weights(task)
    return int(multiples[task.task_name] + 1e-5)

  return 1


def get_task_weights(config, sizes):
  if config.dataset_multiples:
    _, weights = _multiples_and_weights(config)
    return weights
  else:
    if config.task_weight_exponent < 0:
      return {task_name: 1.0 for task_name in sizes}
    n_examples = sum(sizes.values())
    weights = {task_name: 1.0 / (size**(1 - config.task_weight_exponent))
               for task_name, size in sizes.items()}
    expected_weight = sum([weights[task_name] * sizes[task_name] / n_examples
                           for task_name in weights])
    weights = {task_name: w / expected_weight
               for task_name, w in weights.items()}
    return weights

if __name__ == "__main__":
  config = Config(None)
  sizes = {"karolinska": 11250, "radboud": 5538}
  print(get_task_weights(config, sizes))