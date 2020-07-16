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

tile_size = 256
n_tiles = 12

def get_tiles(img, mode=0):
    result = []
    h, w, c = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

    img2 = np.pad(img,[[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2,pad_w - pad_w//2], [0,0]], constant_values=255)
    img3 = img2.reshape(
        img2.shape[0] // tile_size,
        tile_size,
        img2.shape[1] // tile_size,
        tile_size,
        3
    )

    img3 = img3.transpose(0,2,1,3,4).reshape(-1, tile_size, tile_size,3)
    n_tiles_with_info = (img3.reshape(img3.shape[0],-1).sum(1) < tile_size ** 2 * 3 * 255).sum()
    if len(img3) < n_tiles:
        img3 = np.pad(img3,[[0,n_tiles-len(img3)],[0,0],[0,0],[0,0]], constant_values=255)
    idxs = np.argsort(img3.reshape(img3.shape[0],-1).sum(-1))[:n_tiles]
    img3 = img3[idxs]
    for i in range(len(img3)):
        result.append({'img':img3[i], 'idx':i})
    return result, n_tiles_with_info >= n_tiles


def zip_data():
  print(OUT_TRAIN)
  x_tot,x2_tot = [],[]
  names = [name.split(".")[0] for name in os.listdir(TRAIN)]

  for name in tqdm(names):
    img = skimage.io.MultiImage(os.path.join(TRAIN, name + '.tiff'))[1]
    tiles, _ = get_tiles(img)
    for t in tiles:
      img, idx = t['img'],  t['idx']
      cv2.imwrite(os.path.join(OUT_TRAIN, f"{name}_{idx}.png"), img)


if __name__ == "__main__":
  zip_data()