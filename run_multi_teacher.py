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

run_infernce = True
data_dir = os.path.join(os.getcwd(), "data", "train12x128x128")
teacher_dir = "checkpoint/baseline/resnet50_32x4d/epoch=8.ckpt"
# teacher_dir = "checkpoint/ban_student/wide_resnet50_2/epoch=9_v0.ckpt"
device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
  print(os.getcwd())

  for data_provider in ["radboud", "karolinska"]:
    model = TeacherTrainer(data_provider)
    max_epoches = model.get_max_epoches()

    # Load checkpoint
    checkpoint_path = os.path.join(os.getcwd(), "checkpoint", model.model_name, data_provider)
    checkpoint_callback = ModelCheckpoint(
        filepath = checkpoint_path,
        save_top_k = 1,
        verbose = True,
        monitor = 'kappa_score/kappa_score',
        mode = 'max'
    )

    print(">> Save model in %s" % checkpoint_path)

      # Load tensorboard
    tb_logger = loggers.TensorBoardLogger('logs/', name = model.model_name + "/{}".format(data_provider))
    trainer = pl.Trainer(checkpoint_callback = checkpoint_callback,
                        nb_sanity_val_steps = 0,
                        max_epochs = max_epoches,
                        gpus = -1,
                        logger = tb_logger)

    # Assign learning rate
    model.max_lr = 1e-3
    model.configure_optimizers()

    trainer.fit(model)