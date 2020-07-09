import os
import pandas as pd
import numpy as np
from preprocessing.dataset.load_ban_dataset import BanPandaDataset

# Constant value
data_dir = os.path.join(os.getcwd(), "data")
teacher_dir = os.path.join(data_dir, "teacher.json")
random_state = 42
test_sample = 100


def split_dataframe(table):
  test = pd.DataFrame(columns = table.columns)
  for grade in table["isup_grade"].unique():
      test = test.append(table[table["isup_grade"] == grade].sample(n = test_sample, random_state = random_state))

  train = pd.concat([table, test]).drop_duplicates(keep = False)

  assert len(train.merge(test, how = 'inner' ,indicator = False)) == 0
  return train, test


def load_train_data(train_table, transform, data_dir, loss_type, label_type, **kwags):
  # Balance the trainset
  num_sample = train_table.groupby("isup_grade").count()["image_id"].min()
  new_train_table = pd.DataFrame(columns = train_table.columns)
  for grade in train_table["isup_grade"].unique():
    table = train_table[train_table["isup_grade"] == grade]
    new_train_table = new_train_table.append(table.sample(n = num_sample, random_state = 42), ignore_index = True)

  return BanPandaDataset(new_train_table, transform = transform, data_dir = data_dir, loss_type = loss_type, label_type = label_type, **kwags)


def load_test_data(test_table, transform, data_dir, loss_type, label_type):
  return BanPandaDataset(test_table, transform = transform, data_dir = data_dir, loss_type = loss_type, label_type = label_type)


def load_for_ban_student(transform, data_dir, loss_type, label_type, **kwags):
  table = pd.read_csv(teacher_dir, index_col = 0)
  train, test = split_dataframe(table)

  trainset = load_train_data(train, transform[0], data_dir, loss_type, label_type, **kwags)
  testset = load_test_data(test, transform[1], data_dir, loss_type, label_type)
  return trainset, testset