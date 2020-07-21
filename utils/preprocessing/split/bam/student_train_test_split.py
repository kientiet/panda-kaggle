import os
import pandas as pd
import numpy as np
from torch.utils.data.dataset import ConcatDataset
from utils.preprocessing.dataset.bam.load_bam_dataset import BamPandaDataset
from utils.preprocessing.task_weight import task_weighting, config

# Constant value
data_dir = os.path.join(os.getcwd(), "data")
table_dir = os.path.join(os.getcwd(), "data", "train.csv")
teacher_dir = os.path.join(data_dir, "teacher.json")
random_state = 42
test_sample = 100

def adjust_weights(train_table):
  # Adjust the dataframe
  data_providers = train_table["data_provider"].unique()
  new_train_table, sizes = pd.DataFrame(), {}
  for data_provider in data_providers:
    task_config = config.Config(data_provider)
    sub_table = pd.DataFrame()
    for _ in range(task_weighting.get_task_multiple(task_config, split = "train")):
      sub_table = sub_table.append(train_table[train_table["data_provider"] == data_provider], ignore_index = True)

    new_train_table = new_train_table.append(sub_table, ignore_index = True)

    # Get the weights for each tasks
    sizes[data_provider] = len(sub_table)

  weights = task_weighting.get_task_weights(config.Config(None), sizes)
  return new_train_table, weights

def split_table(table, transform, data_dir, loss_type, label_type, **kwags):
  radboud_table, karolinska_table = table[table["data_provider"] == "radboud"], \
      table[table["data_provider"] == "karolinska"]

  radboud_dataset = BamPandaDataset(radboud_table.reset_index(drop = True),
                                    transform = transform,
                                    data_dir = data_dir,
                                    loss_type = loss_type,
                                    label_type = label_type,
                                    task_weighting = kwags["weights"]["radboud"], **kwags)

  karolinska_dataset = BamPandaDataset(karolinska_table.reset_index(drop = True),
                                      transform = transform,
                                      data_dir = data_dir,
                                      loss_type = loss_type,
                                      label_type = label_type,
                                      task_weighting = kwags["weights"]["karolinska"], **kwags)

  return radboud_dataset, karolinska_dataset

def load_train_data(train_table, transform, data_dir, loss_type, label_type, **kwags):
  train_table, weights = adjust_weights(train_table)

  radboud_dataset, karolinska_dataset = split_table(train_table, transform, data_dir, loss_type, label_type, weights = weights, **kwags)
  return ConcatDataset([radboud_dataset, karolinska_dataset])


def load_test_data(test_table, transform, data_dir, loss_type, label_type):
  weights = {
    "karolinska": 1.,
    "radboud": 1.
  }
  radboud_dataset, karolinska_dataset = split_table(test_table, transform, data_dir, loss_type, label_type, weights = weights)

  return ConcatDataset([radboud_dataset, karolinska_dataset])


def load_for_bam_student(train_table, val_table, transform, data_dir, loss_type, label_type, **kwags):
  trainset = load_train_data(train_table, transform[0], data_dir, loss_type, label_type, **kwags)
  testset = load_test_data(val_table, transform[1], data_dir, loss_type, label_type)
  return trainset, testset