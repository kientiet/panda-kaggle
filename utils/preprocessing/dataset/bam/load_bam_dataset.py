import os
import torch
import cv2
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
from ast import literal_eval

from utils.preprocessing.task_weight import config
from utils.preprocessing.dataset.dataset import PandaDataset
from utils.preprocessing.augment.randaug.policy import randaug_policies
from utils.preprocessing.augment.randaug.randaug import apply_policy
from utils.preprocessing.augment.get_transform import get_transform


class BamPandaDataset(PandaDataset):
  def __init__(self, data_frame, transform, data_dir, loss_type, **kwags):
    super().__init__(data_frame, transform, data_dir, loss_type)
    self.label_type = kwags["label_type"]
    self.task_weights = kwags["task_weights"] if "task_weights" in kwags else 1.

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
      transform = self.policy[random.randint(0, len(self.policy) - 1)]

    image_id, number = patient["image_id"], 0
    batch_images = []
    while True:
      temp_dir = os.path.join(img_dir, f"{image_id}_{number}.png")
      if os.path.isfile(temp_dir):
        img = Image.open(temp_dir)

        if self.policy is not None:
          img = self.transform(transform, img)
        else:
          img = self.transform(img)

        batch_images.append(np.array(img))
      else:
        break
      number += 1

    image = cv2.hconcat([cv2.vconcat([batch_images[0], batch_images[1], batch_images[2], batch_images[3]]),
                          cv2.vconcat([batch_images[4], batch_images[5], batch_images[6], batch_images[7]]),
                          cv2.vconcat([batch_images[8], batch_images[9], batch_images[10], batch_images[11]]),
                          cv2.vconcat([batch_images[12], batch_images[13], batch_images[14], batch_images[15]])])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = get_transform(None)(image)

    hard_label, soft_label = patient["isup_grade"], patient["teacher"]
    soft_label = np.array(literal_eval(soft_label))
    if self.label_type == "hard":
      soft_label = np.argmax(soft_label)

    task_weight = (patient["data_provider"], torch.tensor(self.task_weights))

    return image, hard_label, soft_label, *task_weight