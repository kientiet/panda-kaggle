import os
import pandas as pd
import numpy as np

from utils.preprocessing.dataset.load_supervised_dataset import SupervisedDataset

# Constant value
data_dir = os.path.join(os.getcwd(), "data")
dataframe_dir = os.path.join(data_dir, "train.csv")
random_state = 42
test_sample = 100

def split_by_institution(table):
  data_provider = table["data_provider"].unique()
  dataset = {}
  for provider in data_provider:
    dataset[provider] = table[table["data_provider"] == provider]
  return dataset


def split_dataframe(table):
  test = pd.DataFrame(columns = table.columns)
  for grade in table["isup_grade"].unique():
      test = test.append(table[table["isup_grade"] == grade].sample(n = test_sample, random_state = random_state))

  train = pd.concat([table, test]).drop_duplicates(keep = False)

  return train, test

def load_train_data(train_table, transform, data_dir, loss_type, sample):
  # Balance the trainset
  if sample is not None:
    num_sample = train_table.groupby("isup_grade").count()["image_id"].max() if sample == "upsample" else \
      train_table.groupby("isup_grade").count()["image_id"].min()
    replace = sample == "upsample"

    new_train_table = pd.DataFrame(columns = train_table.columns)
    for grade in train_table["isup_grade"].unique():
      table = train_table[train_table["isup_grade"] == grade]
      new_train_table = new_train_table.append(table.sample(n = num_sample, random_state = 42, replace = replace), ignore_index = True)
  else:
    new_train_table = train_table

  return SupervisedDataset(new_train_table, transform = transform, data_dir = data_dir, loss_type = loss_type)


def load_test_data(test_table, transform, data_dir, loss_type):
  return SupervisedDataset(test_table, transform = transform, data_dir = data_dir, loss_type = loss_type)


def load_dataframe(transform, data_dir, loss_type, sample, table = None):
  # Split dataframe to train and test
  if table is None:
    table = pd.read_csv(dataframe_dir)
  train, test = split_dataframe(table)

  assert len(train.merge(test, how = 'inner' ,indicator = False)) == 0

  # Get the dataset class
  trainset = load_train_data(train, transform[0], data_dir, loss_type, sample)
  testset = load_test_data(test, transform[1], data_dir, loss_type)

  return trainset, testset