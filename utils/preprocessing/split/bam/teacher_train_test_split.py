import os
import pandas as pd
import numpy as np
from utils.preprocessing.dataset.supervised.load_dataset import SupervisedDataset

# Constant value
data_dir = os.path.join(os.getcwd(), "data")
table_dir = os.path.join(os.getcwd(), "data", "train.csv")
random_state = 42
test_sample = 50


def split_dataframe(table):
  test = pd.DataFrame(columns = table.columns)
  for grade in table["isup_grade"].unique():
    test = test.append(table[table["isup_grade"] == grade].sample(n = test_sample, random_state = random_state))

  train = pd.concat([table, test]).drop_duplicates(keep = False)

  assert len(train.merge(test, how = 'inner' ,indicator = False)) == 0
  return train, test


def load_train_data(train_table, transform, data_dir, loss_type, **kwags):
  if kwags["sample"] is not None:
    num_sample = train_table.groupby("isup_grade").count()["image_id"].max() if kwags["sample"] == "upsample" else \
      train_table.groupby("isup_grade").count()["image_id"].min()
    replace = kwags["sample"] == "upsample"

    new_train_table = pd.DataFrame(columns = train_table.columns)
    for grade in train_table["isup_grade"].unique():
      table = train_table[train_table["isup_grade"] == grade]
      new_train_table = new_train_table.append(table.sample(n = num_sample, random_state = 42, replace = replace), ignore_index = True)
  else:
    new_train_table = train_table

  return SupervisedDataset(new_train_table, transform = transform, data_dir = data_dir, loss_type = loss_type)


def load_test_data(test_table, transform, data_dir, loss_type):
  return SupervisedDataset(test_table, transform = transform, data_dir = data_dir, loss_type = loss_type)


def load_for_teacher(transform, data_dir, loss_type, data_provider, **kwags):
  print("\n\n>> Load data for %s data_provider" % data_provider)
  table = pd.read_csv(table_dir)
  table = table[table["data_provider"] == data_provider]

  train, test = split_dataframe(table)
  trainset = load_train_data(train, transform[0], data_dir, loss_type, **kwags)
  testset = load_test_data(test, transform[1], data_dir, loss_type)

  return trainset, testset