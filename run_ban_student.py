import os
import copy
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from trainer.distill.ban import BanStudentTrainer

from trainer.supervised.baseline import BaselineTrainer

run_infernce = True
teacher_dir = "checkpoint/baseline/resnet50_32x4d/epoch=6.ckpt"
device = "cuda" if torch.cuda.is_available() else "cpu"

def run_teacher_inference(student):
  print(teacher_dir)
  teacher = BaselineTrainer.load_from_checkpoint(teacher_dir)
  teacher = teacher.to(device)

  trainloader = torch.utils.data.DataLoader(student.trainset, batch_size = 100, shuffle = False, num_workers = 4)
  teacher_inference = []
  with torch.no_grad():
    for images, labels in tqdm(trainloader):
      images, labels = images.to(device), labels.to(device)
      _, logits = teacher.forward(images, labels)
      teacher_inference.append(F.softmax(logits, dim = -1))

      del images
      del labels

  teacher_inference = torch.cat(teacher_inference)
  student.trainset.data_frame["teacher"] = teacher_inference.cpu().numpy().tolist()
  student.trainset.data_frame.to_csv(os.path.join(os.getcwd(), "data", "teacher.json"))
  print(">> Done inference")


if __name__ == "__main__":
  print(os.getcwd())

  model = BanStudentTrainer(-1)
  if run_infernce:
    run_infernce = False
    run_teacher_inference(model)

  # Get necessary parameters
  total_generation = model.stream["generation"]
  max_epoches = model.get_max_epoches()

  for generation in range(total_generation):
    print("\n\n>> Runing the %d generation" % generation)
    model = BanStudentTrainer(generation)
    # Load checkpoint
    checkpoint_path = os.path.join(os.getcwd(), "checkpoint", model.model_name)
    checkpoint_callback = ModelCheckpoint(
        filepath = checkpoint_path,
        save_top_k = 5,
        verbose = True,
        monitor = 'kappa_score',
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

    # Assign learning rate
    model.max_lr = 3e-2
    model.configure_optimizers()

    trainer.fit(model)

    # Re-assign the model
    best_models = checkpoint_callback.best_k_models
    max_kappa_score, teacher_dir = -100., ""
    for key, value in best_models.items():
      if max_kappa_score < best_models[key]:
        max_kappa_score = best_models[key]
        teacher_dir = str(key)

    print(">> Load the best model %s" % teacher_dir)
    model = model.load_from_checkpoint(teacher_dir)
    model.run_inference()
