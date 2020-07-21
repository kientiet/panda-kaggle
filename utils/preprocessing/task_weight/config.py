import os

class Config(object):
  def __init__(self, task_name):
    super().__init__()
    # Task information
    self.task_name = task_name

    # Sample config
    self.task_weight_exponent = 0.75
    self.dataset_multiples = True