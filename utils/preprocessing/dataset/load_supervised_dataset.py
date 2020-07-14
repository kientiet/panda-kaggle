import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict

from utils.preprocessing.augment.get_transform import get_transform
from utils.preprocessing.dataset.dataset import PandaDataset

class SupervisedDataset(PandaDataset):
  def __init__(self, data_frame, transform, data_dir, loss_type):
    super().__init__(data_frame, transform, data_dir, loss_type)


  def __getitem__(self, index):
    patient = self.data_frame.iloc[index]
    img_dir = os.path.join(os.getcwd(), "data", self.data_dir)

    image_id, number = patient["image_id"], 0
    batch_images = []
    while True:
      temp_dir = os.path.join(img_dir, f"{image_id}_{number}.png")
      if os.path.isfile(temp_dir):
        img = Image.open(temp_dir).convert("RGB")

        # Convert before transform
        np_img = 255. - np.array(img)
        img = Image.fromarray(np.uint8(np_img)).convert("RGB")

        img = self.transform(img)
        batch_images.append(img)
      else:
        break
      number += 1
    batch_images = torch.stack(batch_images)

    label = patient["isup_grade"]
    if self.loss_type == "bceloss":
      bin_label = np.zeros(5)
      bin_label[:label] = 1
      return batch_images, bin_label
    else:
      return batch_images, label