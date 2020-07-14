import os
import torch
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
from ast import literal_eval

from utils.preprocessing.dataset.dataset import PandaDataset
from utils.preprocessing.augment.randaug.policy import randaug_policies
from utils.preprocessing.augment.randaug.randaug import apply_policy

class OutliersPandaDataset(PandaDataset):
  def __init__(self, data_frame, transform, data_dir, loss_type, **kwags):
    super().__init__(data_frame, transform, data_dir, loss_type)
    self.policy = None
    self.policy = randaug_policies()
    self.transform = apply_policy


  def __getitem__(self, index):
    patient = self.data_frame.iloc[index]
    img_dir = os.path.join(os.getcwd(), "data", self.data_dir)

    ## Apply the same transformation on all picture
    transform = self.policy[random.randint(0, len(self.policy) - 1)]

    image_id, number = patient["image_id"], 0
    student_batch_images = []
    while True:
      temp_dir = os.path.join(img_dir, f"{image_id}_{number}.png")
      if os.path.isfile(temp_dir):
        img = Image.open(temp_dir)
        img = self.transform(transform, img)
        student_batch_images.append(img)
      else:
        break
      number += 1

    student_batch_images = torch.stack(student_batch_images)

    hard_label = patient["isup_grade"]

    return student_batch_images, hard_label