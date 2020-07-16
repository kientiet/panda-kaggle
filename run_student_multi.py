import os
import copy
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from tqdm import tqdm
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from trainer.distill.multil_teacher import TeacherTrainer
from trainer.distill.student_multi import StudentMultiTrainer

run_infernce = False
data_dir = os.path.join(os.getcwd(), "data", "train12x128x128")
train_save = os.path.join(os.getcwd(), "data", "multi_train.json")
val_save = os.path.join(os.getcwd(), "data", "multi_val.json")

radboud_teacher_dir = ["checkpoint/multi/resnet18/radboud/version1/epoch=13_v0.ckpt", "radboud"]
karolinska_teacher_dir = ["checkpoint/multi/resnet18/karolinska/version1/epoch=13_v0.ckpt", "karolinska"]

device = "cuda" if torch.cuda.is_available() else "cpu"


def run_teacher_infernce():
  train_table, val_table = pd.DataFrame(), pd.DataFrame()
  for teacher_dir in [radboud_teacher_dir, karolinska_teacher_dir]:
    teacher_dir, data_provider= teacher_dir
    teacher = TeacherTrainer.load_from_checkpoint(teacher_dir, data_provider = data_provider)
    teacher = teacher.to(device)
    teacher.eval()
    with torch.no_grad():
      preds = []
      for images, labels in tqdm(teacher.trainloader):
        images, labels = images.to(device), labels.to(device)
        _, logits = teacher.forward(images, labels)
        preds.append(F.softmax(logits, dim = -1))

      preds = torch.cat(preds)
      teacher.trainset.data_frame["teacher"] = preds.cpu().numpy().tolist()
      train_table = train_table.append(teacher.trainset.data_frame, ignore_index = True)


      preds = []
      for images, labels in tqdm(teacher.valloader):
        images, labels = images.to(device), labels.to(device)
        _, logits = teacher.forward(images, labels)
        preds.append(F.softmax(logits, dim = -1))

      preds = torch.cat(preds)
      teacher.valset.data_frame["teacher"] = preds.cpu().numpy().tolist()
      val_table = val_table.append(teacher.valset.data_frame, ignore_index = True)

  return train_table, val_table

if __name__ == "__main__":
  print(os.getcwd())
  if run_infernce:
    train_table, val_table = run_teacher_infernce()
    train_table.to_csv(train_save)
    val_table.to_csv(val_save)
  else:
    train_table = pd.read_csv(train_save, index_col = 0)
    val_table = pd.read_csv(val_save, index_col = 0)

  model = StudentMultiTrainer(train_table, val_table)
  max_epoches = model.get_max_epoches()

  # Load checkpoint
  checkpoint_path = os.path.join(os.getcwd(), "checkpoint", model.model_name)
  checkpoint_callback = ModelCheckpoint(
      filepath = checkpoint_path,
      save_top_k = 5,
      verbose = True,
      monitor = 'kappa_score/kappa_score',
      mode = 'max'
  )

  print(">> Save model in %s" % checkpoint_path)

    # Load tensorboard
  tb_logger = loggers.TensorBoardLogger('logs/', name = model.model_name)
  trainer = pl.Trainer(checkpoint_callback = checkpoint_callback,
                      nb_sanity_val_steps = 0,
                      max_epochs = max_epoches,
                      gpus = -1,
                      logger = tb_logger)

  trainer.fit(model)