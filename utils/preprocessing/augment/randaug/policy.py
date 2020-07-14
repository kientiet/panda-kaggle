from utils.preprocessing.augment.randaug import randaug
import torchvision.transforms as transforms

def get_trans_list():
  return list(randaug.TRANSFORM_NAMES)


def randaug_policies():
  trans_list = get_trans_list()
  op_list = []
  for trans in trans_list:
    for magnitude in range(1, 4):
      op_list += [(trans, 0.5, magnitude)]

  policies = []
  for op_1 in op_list:
    for op_2 in op_list:
      policies += [[op_1, op_2]]

  return policies