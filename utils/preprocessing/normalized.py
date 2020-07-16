import staintools
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

data_dir = os.path.join(os.getcwd(), "data", "train12x128x128")
des_dir = os.path.join(os.getcwd(), "data", "normalized")

if __name__ == "__main__":

  ## Load the first picture
  all_files = os.listdir(data_dir)
  target = staintools.read_image(os.path.join(data_dir, all_files[0]))
  target = staintools.LuminosityStandardizer.standardize(target)

  normalizer = staintools.StainNormalizer(method = "vahadane")
  normalizer.fit(target)

  if not os.path.isdir(des_dir):
    os.mkdir(des_dir)

  for image_dir in tqdm(all_files):
    to_transform = staintools.read_image(os.path.join(data_dir, image_dir))
    transformed = normalizer.transform(to_transform)
    transformed = Image.fromarray(transformed).convert("RGB")
    transformed.save(os.path.join(des_dir, image_dir))

