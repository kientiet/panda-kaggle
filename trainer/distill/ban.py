import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import pytorch_lightning as pl
from tqdm import tqdm
import copy
from preprocessing.split.ban_train_test_split import load_for_ban_student
from trainer.distill.trainer import DistillTrainerSkeleton
from trainer.distill.teacher import TeacherTrainer
from models.resnet import ResNetModel
from models.project_layer import ProjectLayer

config_dir = os.path.join(os.getcwd(), "config", "ban_student.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"

class BanStudentTrainer(DistillTrainerSkeleton):
  def __init__(self, current_generation):
    super(DistillTrainerSkeleton, self).__init__(config_dir)
    ## Init for Ban Student
    self.generation = self.stream["generation"]
    self.current_generation = current_generation
    self.reset_model(self.stream)


  def run_inference(self):
    print("\n\n>> Start runing the inference")
    trainloader = torch.utils.data.DataLoader(self.trainset, batch_size = 100, shuffle = False, num_workers = 4)
    teacher_inference = []
    with torch.no_grad():
      for images, labels in tqdm(trainloader):
        _, logits = self.forward(images, labels, None, running_mode = "validation")
        teacher_inference.append(F.softmax(logits, dim = -1))

        del images
        del labels

    teacher_inference = torch.cat(teacher_inference)
    self.trainset.data_frame["teacher"] = teacher_inference.cpu().numpy().tolist()
    self.trainset.data_frame.to_csv(os.path.join(os.getcwd(), "data", "teacher.json"))
    print(">> Done inference")


  def reset_model(self, stream = None):
    # Reload dataset
    self.student_trainset, self.valset = load_for_ban_student(transform = [self.stream["student_transformation"], self.stream["test_transformation"]], \
                                                            data_dir = self.stream["data_dir"],
                                                            loss_type = self.stream["loss_func"],
                                                            label_type = self.stream["label_type"],
                                                            difficulty = self.stream["difficulty"][self.current_generation]
                                                            )

    print("\n\n>> The transformation of student is")
    print(self.student_trainset.transform)
    self.trainloader = torch.utils.data.DataLoader(self.student_trainset, batch_size = self.stream["batch_size"], shuffle = True, num_workers = 4)
    self.valloader = torch.utils.data.DataLoader(self.valset, batch_size = self.stream["batch_size"], shuffle = False, num_workers = 4)

    # Update the dataloader
    self.train_dataloader()
    self.val_dataloader()

    # Init student encoder and project_layer
    self.encoder = ResNetModel(self.stream["backbone"], pretrained = self.stream["pretrained"], \
      stochastic_depth_prob = self.stream["stochastic_depth_prob"])
    self.project_layer = ProjectLayer(self.encoder.num_channel, self.stream["num_classes"])

    # Optimzer and scheduler
    self.configure_optimizers(stream)

    # Init the loss function for student
    self.loss_func = self.loss_func(self.stream["anneal_type"], coeff = self.stream["coeff"], temperature = self.stream["temperature"], \
      total_step = self.get_total_step())


  def forward(self, images, hard_labels, soft_labels, running_mode):
    # Prepare the batch
    batch_size, image_batch, channels, height, width = images.shape
    images = images.reshape(-1, channels, height, width)

    # Forward pass for student
    logits = self.encoder(images)
    shape = logits.shape
    # concatenate the output for tiles into a single map
    logits = logits.view(-1, image_batch, shape[1], shape[2], shape[3]).permute(0, 2, 1, 3, 4).contiguous() \
      .view(-1, shape[1], shape[2] * image_batch, shape[3])
    logits = self.project_layer(logits)

    if running_mode == "training":
      loss, hard_loss, soft_loss, coeff = self.loss_func(logits, hard_labels, soft_labels, label_type = self.stream["label_type"], running_mode = running_mode)
      return loss, hard_loss, soft_loss, coeff
    else:
      loss = self.loss_func(logits, hard_labels, soft_labels, label_type = self.stream["label_type"], running_mode = running_mode)
      return loss, logits


  def training_step(self, train_batch, batch_idx):
    student_images, hard_labels, soft_labels = train_batch
    loss, hard_loss, soft_loss, coeff = self.forward(student_images, hard_labels, soft_labels, "training")
    logs = {"learning_rate": self.optimizer.param_groups[0]["lr"],
            "train_loss": loss.item(),
            "hard_loss": hard_loss.item(),
            "soft_loss": soft_loss.item(),
            "coeff": coeff}
    return {"loss": loss, "log": logs}


  def validation_step(self, val_batch, val_idx):
    student_images, hard_labels, soft_labels = val_batch
    val_loss, logits  = self.forward(student_images, hard_labels, soft_labels, "validation")

    preds = torch.max(F.softmax(logits, dim = -1), dim = -1)[1]
    logs = {"val_loss": val_loss,
            "y_pred": preds.cpu().numpy()}

    return logs