import torch
import torchvision.transforms as transforms

standard_transforms = transforms.Compose([
  transforms.RandomVerticalFlip(p = 0.5),
  transforms.RandomHorizontalFlip(p = 0.5),
  transforms.RandomRotation(15),
  transforms.ToTensor(),
  transforms.Normalize([0.90949707, 0.8188697, 0.87795304], [0.36357649, 0.49984502, 0.40477625])
  # transforms.Normalize([1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304], [0.36357649, 0.49984502, 0.40477625])
])


no_transforms = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize([0.90949707, 0.8188697, 0.87795304], [0.36357649, 0.49984502, 0.40477625])
  # transforms.Normalize([1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304], [0.36357649, 0.49984502, 0.40477625])
])


transform_dict = {
  None: no_transforms,
  "standard_transforms": standard_transforms,
  "student_transformation": None
}

def get_transform(transform_name):
  return transform_dict[transform_name]