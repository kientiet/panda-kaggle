import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.preprocessing.augment.get_transform import get_transform
from utils.preprocessing.dataset.dataset import PandaDataset


class SupervisedDataset(PandaDataset):
  def __init__(self, data_frame, transform, data_dir, loss_type, **kwags):
    super().__init__(data_frame, transform, data_dir, loss_type)

  def __getitem__(self, index):
    patient = self.data_frame.iloc[index]
    img_dir = os.path.join(os.getcwd(), "data", self.data_dir)

    transform = None
    if self.policy is not None:
      transform = self.policy[random.randint(0, len(self.policy) - 1)]

    image_id, number = patient["image_id"], 0
    batch_images = []
    while True:
      temp_dir = os.path.join(img_dir, f"{image_id}_{number}.png")
      if os.path.isfile(temp_dir):
        img = Image.open(temp_dir).convert("RGB")

        # Convert before transform
        np_img = 255. - np.array(img)
        img = Image.fromarray(np.uint8(np_img)).convert("RGB")

        if transform is not None:
          img = self.transform(transform, img)
        elif self.transform is not None:
          img = self.transform(img)

        batch_images.append(np.array(img))
      else:
        break
      number += 1

    # Testing for 16x128x128
    image = cv2.hconcat([cv2.vconcat([batch_images[0], batch_images[1], batch_images[2], batch_images[3]]),
                          cv2.vconcat([batch_images[4], batch_images[5], batch_images[6], batch_images[7]]),
                          cv2.vconcat([batch_images[8], batch_images[9], batch_images[10], batch_images[11]]),
                          cv2.vconcat([batch_images[12], batch_images[13], batch_images[14], batch_images[15]])])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = get_transform(None)(image)

    label = patient["isup_grade"]
    if self.loss_type == "bceloss":
      bin_label = np.zeros(5)
      bin_label[:label] = 1
      return image, bin_label
    else:
      return image, label