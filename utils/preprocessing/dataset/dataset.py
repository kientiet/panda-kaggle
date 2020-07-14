import os
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.preprocessing.augment.get_transform import get_transform

class PandaDataset(Dataset):
  def __init__(self, data_frame, transform, data_dir, loss_type):
    super().__init__()
    self.data_frame = data_frame
    self.transform = get_transform(transform)
    self.data_dir = data_dir
    self.loss_type = loss_type


  def __getitem__(self, index):
    pass


  def get_test_label(self):
    return self.data_frame["isup_grade"].values.astype(int), self.data_frame


  def change_transform(self, transform):
    self.transform = get_transform(transform)


  def __len__(self):
    return len(self.data_frame)