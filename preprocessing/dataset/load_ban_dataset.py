import os
import torch
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
from ast import literal_eval

from preprocessing.dataset.dataset import PandaDataset
from preprocessing.augment.randaug.policy import randaug_policies
from preprocessing.augment.randaug.randaug import apply_policy

class BanPandaDataset(PandaDataset):
  def __init__(self, data_frame, transform, data_dir, loss_type, **kwags):
    super().__init__(data_frame, transform, data_dir, loss_type)
    self.label_type = kwags["label_type"]
    if "difficulty" in kwags:
      self.difficulty = kwags["difficulty"]

    self.policy = None
    if transform == "student_transformation":
      self.policy = randaug_policies(self.difficulty)
      self.transform = apply_policy


  def __getitem__(self, index):
    patient = self.data_frame.iloc[index]
    img_dir = os.path.join(os.getcwd(), "data", self.data_dir)

    ## Apply the same transformation on all picture
    transform = self.transform
    if self.policy is not None:
      transform = self.policy[random.randint(0, len(self.policy))]

    image_id, number = patient["image_id"], 0
    student_batch_images = []
    while True:
      temp_dir = os.path.join(img_dir, f"{image_id}_{number}.png")
      if os.path.isfile(temp_dir):
        img = Image.open(temp_dir)

        if self.policy is not None:
          img = self.transform(transform, img)

        student_batch_images.append(img)
      else:
        break
      number += 1

    student_batch_images = torch.stack(student_batch_images)

    hard_label, soft_label = patient["isup_grade"], patient["teacher"]
    soft_label = np.array(literal_eval(soft_label))
    if self.label_type == "hard":
      soft_label = np.argmax(soft_label)

    return student_batch_images, hard_label, soft_label