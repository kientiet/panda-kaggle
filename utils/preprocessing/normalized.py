import staintools
import os
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from joblib import Parallel, delayed
import logging

data_dir = os.path.join(os.getcwd(), "data", "train12x128x128")
des_dir = os.path.join(os.getcwd(), "data", "normalized")


logging.basicConfig(level=logging.DEBUG,
                    format='%(message)s %(threadName)s %(processName)s',
                    )

def stain_normalization(image_dir):
  # logging.debug("debug logging: ")
  to_transform = staintools.read_image(os.path.join(data_dir, image_dir))
  try:
    transformed = normalizer.transform(to_transform)
  except staintools.miscellaneous.exceptions.TissueMaskException:
    transformed = to_transform

  transformed = Image.fromarray(transformed).convert("RGB")
  transformed.save(os.path.join(des_dir, image_dir))


if __name__ == "__main__":

  ## Load the first picture
  all_files = os.listdir(data_dir)
  target = staintools.read_image(os.path.join(data_dir, all_files[0]))
  target = staintools.LuminosityStandardizer.standardize(target)

  normalizer = staintools.StainNormalizer(method = "vahadane")
  normalizer.fit(target)

  if not os.path.isdir(des_dir):
    os.mkdir(des_dir)

  processed_list = Parallel(n_jobs = -1, prefer="processes")(delayed(stain_normalization)(image_dir) \
                                                    for image_dir in tqdm(all_files))