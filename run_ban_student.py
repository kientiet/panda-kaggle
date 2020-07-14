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

from utils.preprocessing.dataset.load_supervised_dataset import SupervisedDataset
from trainer.distill.ban import BanStudentTrainer
from trainer.supervised.baseline import BaselineTrainer
from sklearn.metrics import accuracy_score, cohen_kappa_score

run_infernce = True
data_dir = os.path.join(os.getcwd(), "data", "train12x128x128")
teacher_dir = "checkpoint/baseline/resnet50_32x4d/epoch=8.ckpt"
# teacher_dir = "checkpoint/ban_student/wide_resnet50_2/epoch=9_v0.ckpt"
device = "cuda" if torch.cuda.is_available() else "cpu"

def run_teacher_inference(trainer, teacher_dir, current_generation):
  print("\n\n>> Load from %s" % teacher_dir)
  if current_generation is not None:
    teacher = trainer.load_from_checkpoint(teacher_dir, current_generation = current_generation)
  else:
    teacher = trainer.load_from_checkpoint(teacher_dir)

  teacher = teacher.to(device)

  combine = pd.read_csv("data/train.csv")
  dataset = SupervisedDataset(combine, None, data_dir, "crossentropy")
  trainloader = torch.utils.data.DataLoader(dataset, batch_size = 100, shuffle = False)
  print(">> The total lenght is %d" % len(trainloader))
  teacher_inference = []

  y_pred, y_true = np.array([]), np.array([])
  teacher.eval()
  with torch.no_grad():
    for images, labels in tqdm(trainloader):
      images, labels = images.to(device), labels.to(device)
      if current_generation is not None:
        _, logits = teacher.forward(images, labels, None, "validation")
      else:
        _, logits = teacher.forward(images, labels)
      teacher_inference.append(F.softmax(logits, dim = -1))

      y_true = np.append(y_true, labels.cpu().numpy())
      logits = torch.max(F.softmax(logits, dim = -1), dim = -1)[1]
      y_pred = np.append(y_pred, logits.cpu().numpy())

      del images
      del labels

  print("Kappa score is about %.4f" % cohen_kappa_score(y_true, y_pred, weights = "quadratic"))
  teacher_inference = torch.cat(teacher_inference)
  combine["teacher"] = teacher_inference.cpu().numpy().tolist()
  combine.to_csv(os.path.join(os.getcwd(), "data", "teacher.json"))
  print(">> Done inference and save to %s" % os.path.join(os.getcwd(), "data", "teacher.json"))
  print("=" * 100)


if __name__ == "__main__":
  print(os.getcwd())

  model = BanStudentTrainer(-1)
  if run_infernce:
    run_infernce = False
    # run_teacher_inference(BanStudentTrainer, teacher_dir, 0)
    run_teacher_inference(BaselineTrainer, teacher_dir, None)

  # Get necessary parameters
  total_generation = model.stream["generation"]
  max_epoches = model.get_max_epoches()

  for generation in range(total_generation):
    print("\n\n>> Runing the %d generation" % generation)
    model = BanStudentTrainer(generation)
    # Load checkpoint
    checkpoint_path = os.path.join(os.getcwd(), "checkpoint", model.model_name, "gen_{}".format(generation))
    checkpoint_callback = ModelCheckpoint(
        filepath = checkpoint_path,
        save_top_k = 5,
        verbose = True,
        monitor = 'kappa_score/kappa_score',
        mode = 'max'
    )

    print(">> Save model in %s" % checkpoint_path)

    # Load tensorboard
    tb_logger = loggers.TensorBoardLogger('logs/', name = model.model_name + "/gen_{}".format(generation))
    trainer = pl.Trainer(checkpoint_callback = checkpoint_callback,
                        nb_sanity_val_steps = 0,
                        max_epochs = max_epoches,
                        gpus = -1,
                        logger = tb_logger)

    # Assign learning rate
    model.max_lr = 1e-3
    model.configure_optimizers()
    print(model.trainset.transform)

    trainer.fit(model)

    # Re-assign the model
    best_models = checkpoint_callback.best_k_models
    max_kappa_score, teacher_dir = -100., ""
    for key, value in best_models.items():
      if max_kappa_score < best_models[key]:
        max_kappa_score = best_models[key]
        teacher_dir = str(key)

    print(">> Load the best model %s" % teacher_dir)
    run_teacher_inference(BanStudentTrainer, teacher_dir, generation)
