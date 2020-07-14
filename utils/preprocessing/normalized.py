import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from augment.stain_normalization.stainNorm_Reinhard import Normalizer
from augment.stain_normalization.stain_utils import read_image

data_dir = os.path.join(os.getcwd(), "data", "train12x128x128")
des_dir = os.path.join(os.getcwd(), "data", "normalized")

if __name__ == "__main__":
  normalizer = Normalizer()

  ## Load the first picture
  all_files = os.listdir(data_dir)
  first_picture = read_image(os.path.join(data_dir, all_files[12]))
  normalizer.fit(first_picture)

  if not os.path.isdir(des_dir):
    os.mkdir(des_dir)

  fig, (ax1, ax2) = plt.subplots(1, 2)
  for image_dir in tqdm(all_files):
    image = read_image(os.path.join(data_dir, image_dir))
    image_transform = normalizer.transform(image)
    image = Image.fromarray(image_transform).convert("RGB")
    image.save(os.path.join(des_dir, image_dir))

