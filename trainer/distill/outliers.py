import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.preprocessing.split.outliers_train_test_split import load_for_outliers
from trainer.distill.trainer import DistillTrainerSkeleton

# Model Architecture
from models.resnet import ResNetModel
from models.project_layer import ProjectLayer

# Teacher Architecture
from trainer.supervised.baseline import BaselineTrainer

config_dir = os.path.join(os.getcwd(), "config", "outliers.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"

class OutliersTrainer(DistillTrainerSkeleton):
  def __init__(self):
    super(OutliersTrainer, self).__init__(config_dir)
    ## Init for Ban Student
    self.init_model(self.stream)

  def init_model(self, stream = None):
    # Reload dataset
    self.trainset, self.valset = load_for_outliers(transform = [self.stream["student_aug"], self.stream["teacher_aug"], self.stream["test_aug"]], \
                                                      data_dir = self.stream["data_dir"],
                                                      loss_type = self.stream["loss_func"],
                                                      sample = self.stream["sample"]
                                                      )

    print("\n\n>> The transformation of student is")
    print(self.trainset.transform)
    self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size = self.stream["batch_size"], shuffle = True, num_workers = 4)
    self.valloader = torch.utils.data.DataLoader(self.valset, batch_size = self.stream["batch_size"], shuffle = False, num_workers = 4)

    # Update the dataloader
    self.train_dataloader()
    self.val_dataloader()

    self.config_model()


  def config_model(self):
    # Init student encoder and project_layer
    self.encoder = ResNetModel(self.stream["backbone"], pretrained = self.stream["pretrained"], \
      stochastic_depth_prob = self.stream["stochastic_depth_prob"])
    self.project_layer = ProjectLayer(self.encoder.num_channel, self.stream["num_classes"])

    # Load teacher
    # self.teacher = BaselineTrainer.load_from_checkpoint(self.stream["teacher_list"])
    self.teacher = BaselineTrainer()
    self.teacher.eval()

    # Optimzer and scheduler
    self.configure_optimizers(self.stream)

    # Init the loss function for student
    self.loss_func = self.loss_func(self.stream["anneal_type"], coeff = self.stream["coeff"], temperature = self.stream["temperature"], \
      total_step = self.get_total_step(), keep_classes = self.stream["keep_classes"])


  def forward(self, student_images, teacher_images, hard_labels, running_mode):
    # Prepare the batch
    batch_size, image_batch, channels, height, width = student_images.shape

    if running_mode == "training":
      ## Forward pass for teacher
      with torch.no_grad():
        _, soft_logits = self.teacher.forward(student_images, hard_labels)

    student_images = student_images.reshape(-1, channels, height, width)
    ## Forward pass for student
    logits = self.encoder(student_images)
    shape = logits.shape
    # concatenate the output for tiles into a single map
    logits = logits.view(-1, image_batch, shape[1], shape[2], shape[3]).permute(0, 2, 1, 3, 4).contiguous() \
      .view(-1, shape[1], shape[2] * image_batch, shape[3])
    logits = self.project_layer(logits)


    if running_mode == "training":
      loss, hard_loss, soft_loss, coeff = self.loss_func(logits, hard_labels, soft_logits, running_mode = running_mode)
      return loss, hard_loss, soft_loss, coeff
    else:
      loss = self.loss_func(logits, hard_labels, None, running_mode = running_mode)
      return loss, logits


  def training_step(self, train_batch, batch_idx):
    student_images, teacher_images, hard_labels = train_batch
    loss, hard_loss, soft_loss, coeff = self.forward(student_images, teacher_images, hard_labels, "training")
    logs = {"learning_rate": self.optimizer.param_groups[0]["lr"],
            "train_loss": loss.item(),
            "hard_loss": hard_loss.item(),
            "soft_loss": soft_loss.item(),
            "coeff": coeff
            }

    return {"loss": loss, "log": logs}


  def validation_step(self, val_batch, val_idx):
    student_images, hard_labels = val_batch
    val_loss, logits  = self.forward(student_images, None, hard_labels, "validation")

    preds = torch.max(F.softmax(logits, dim = -1), dim = -1)[1]
    logs = {"val_loss": val_loss,
            "y_pred": preds.cpu().numpy()}

    return logs