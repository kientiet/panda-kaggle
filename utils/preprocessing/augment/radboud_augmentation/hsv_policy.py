import random
import numpy as np
import PIL
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

class RandomRotationWithProb:
  def __init__(self, p = 1):
    self.p = p

  def __call__(self, img):
    p = random.uniform(0, 1)
    if p > self.p:
      img = TF.rotate(img, 90)

    return img


class ElasticDeformation:
  def __init__(self, alpha: tuple = (80, 120), sigma: tuple = (9.0, 11.0), p = 0.5):
    self.alpha = alpha
    self.sigma = sigma
    self.p = p

  def __call__(self, img):
    if random.uniform(0, 1) > self.p:
      img = np.array(img)
      ## ? Random hyperparameters
      shape = img.shape
      alpha, sigma = random.uniform(*self.alpha), random.uniform(*self.sigma)

      random_state = np.random.RandomState(None)

      dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
      dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
      dz = np.zeros_like(dx)

      x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
      indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

      distored_image = map_coordinates(img, indices, order=1, mode='reflect')
      img = Image.fromarray(distored_image.reshape(img.shape)).convert("RGB")

    return img


class AdditiveGaussianNoise:
  def __init__(self, sigma: tuple = (0, 0.1), p = 0.5):
    self.sigma = sigma
    self.p = p

  def __call__(self, img):
    if random.uniform(0, 1) > self.p:
      # PIL to numpy
      img = np.array(img)

      mean = 0.0
      std = random.uniform(*self.sigma)
      noisy_img = img + np.random.normal(mean, std, img.shape)
      img = np.clip(noisy_img, 0, 255)

      img = Image.fromarray(np.uint8(img)).convert("RGB")
    return img


class GaussianBlur:
  def __init__(self, sigma: tuple = (0, 0.1), p = 0.5):
    self.sigma = sigma
    self.p = p

  def __call__(self, img):
    if random.uniform(0, 1) > self.p:
      sigma = random.uniform(*self.sigma)
      radius = sigma * np.sqrt(2 * np.log(225.)) - 1

      img = img.filter(ImageFilter.GaussianBlur(radius = radius))

    return img


def basic(img, p = 0.5):
  if random.uniform(0, 1) > p:
    return transforms.Compose([
      RandomRotationWithProb(0.5),
      transforms.Pad(padding = 4, padding_mode = "reflect")
    ])(img)

  return img

def morphology(img, p = 0.5):
  if random.uniform(0, 1) > p:
    return transforms.Compose([
      transforms.RandomResizedCrop(128, scale = (0.8, 1.2)),
      ElasticDeformation(alpha = (80, 120), sigma = (9.0, 11.0)),
      AdditiveGaussianNoise(sigma = (0, 0.1)),
      GaussianBlur(sigma = (0, 0.1))
    ])(img)

  return img


def bc_transform(img, p = 0.5):
  if random.uniform(0, 1) > p:
    return transforms.Compose([
      transforms.ColorJitter(brightness = (0.65, 1.35), contrast = (0.5, 1.5))
    ])(img)

  return img


def hsv_light(img, p = 0.5):
  img = basic(img, p = 0.5)
  img = morphology(img, p = 0.5)
  img = bc_transform(img, p = 0.5)

  if random.uniform(0, 1) > p:
    img = TF.adjust_saturation(img, random.uniform(-0.5, 0.5))
    img = TF.adjust_hue(img, random.uniform(-0.5, 0.5))

  return img

# def hsv_strong(img):
#   img = basic(img)
#   img = morphology(img)
#   img = bc_transform(img)

#   img = TF.adjust_saturation(img, random.uniform(-1.0, 1.0))
#   img = TF.adjust_hue(img, random.uniform(-1.0, 1.0))
#   return hsv_aug(img)