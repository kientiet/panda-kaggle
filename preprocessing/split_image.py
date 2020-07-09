import os
import cv2
import skimage.io
from tqdm import tqdm
import zipfile
import numpy as np

TRAIN = os.path.join(os.getcwd(), "data", "prostate-cancer-grade-assessment", "train_images")
MASKS = os.path.join(os.getcwd(), "data", "prostate-cancer-grade-assessment", "train_label_masks")
OUT_TRAIN = os.path.join(os.getcwd(), "data", "train")
OUT_MASKS = os.path.join(os.getcwd(), "data", "mask")
sz = 256
N = 36


def tile(img):
    result = []
    shape = img.shape
    pad0,pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz

    img = np.pad(img, [[pad0 // 2,pad0 - pad0 // 2], [pad1 // 2,pad1 - pad1 // 2], [0, 0]],
                constant_values = 255)
    img = img.reshape(img.shape[0] // sz, sz, img.shape[1] //sz, sz, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)

    if len(img) < N:
      img = np.pad(img, [[0, N - len(img)],[0, 0], [0, 0], [0, 0]] ,constant_values = 255)

    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:N]
    img = img[idxs]

    for i in range(len(img)):
      result.append({'img': img[i],
          'idx':i})
    return result


def zip_data():
  print(OUT_TRAIN)
  x_tot,x2_tot = [],[]
  names = [name.split(".")[0] for name in os.listdir(TRAIN)]

  for name in tqdm(names):
    img = skimage.io.MultiImage(os.path.join(TRAIN, name + '.tiff'))[-1]
    # mask = skimage.io.MultiImage(os.path.join(MASKS, name + '_mask.tiff'))[-1]
    tiles = tile(img)

    for t in tiles:
      img, idx = t['img'],  t['idx']

      x_tot.append((img / 255.0).reshape(-1, 3).mean(0))
      x2_tot.append(((img / 255.0) ** 2).reshape(-1, 3).mean(0))

      #if read with PIL RGB turns into BGR
      # img = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]
      cv2.imwrite(os.path.join(OUT_TRAIN, f"{name}_{idx}.png"), img)
      # mask = cv2.imencode('.png', mask[:, :, 0])[1]
      # cv2.imwrite(os.path.join(OUT_MASKS, f"{name}_{idx}.png"), mask)


if __name__ == "__main__":
  zip_data()