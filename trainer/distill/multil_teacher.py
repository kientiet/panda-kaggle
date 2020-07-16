import os
import torch
import pytorch_lightning as pl
from trainer.distill.trainer import DistillTrainerSkeleton

from utils.preprocessing.split.bam_train_test_split import load_for_teacher
from models.resnet import ResNetModel
from models.project_layer import ProjectLayer

config_dir = os.path.join(os.getcwd(), "config", "distill_teacher.yaml")

class TeacherTrainer(DistillTrainerSkeleton):
  def __init__(self, data_provider, trainset = None, valset = None):
    super(TeacherTrainer, self).__init__(yaml_file = config_dir, trainset = trainset, valset = valset)
    self.data_provider = data_provider
    self.init_model()


  def init_model(self):
    self.trainset, self.valset = load_for_teacher(transform = [self.stream["train_aug"], self.stream["test_aug"]], \
                                                  data_dir = self.stream["data_dir"],
                                                  loss_type = self.stream["loss_func"],
                                                  sample = self.stream["sample"],
                                                  data_provider = self.data_provider
                                                  )

    print("\n\n>> The transformation of teacher is")
    print(self.trainset.transform)
    self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size = self.stream["batch_size"], shuffle = True, num_workers = 4)
    self.valloader = torch.utils.data.DataLoader(self.valset, batch_size = self.stream["batch_size"], shuffle = False, num_workers = 4)

    # Update the dataloader
    self.train_dataloader()
    self.val_dataloader()

    self.encoder = ResNetModel(self.stream["backbone"], pretrained = self.stream["pretrained"], \
      stochastic_depth_prob = self.stream["stochastic_depth_prob"])
    self.project_layer = ProjectLayer(self.encoder.num_channel, self.stream["num_classes"])

    # Optimzer and scheduler
    self.configure_optimizers(self.stream)


  def forward(self, images, labels):
    # Prepare the batch
    batch_size, image_batch, channels, height, width = images.shape
    images = images.reshape(-1, channels, height, width)

    # Forward pass
    logits = self.encoder(images)
    shape = logits.shape
    # concatenate the output for tiles into a single map
    logits = logits.view(-1, image_batch, shape[1], shape[2], shape[3]).permute(0, 2, 1, 3, 4).contiguous() \
      .view(-1, shape[1], shape[2] * image_batch, shape[3])
    logits = self.project_layer(logits)
    return self.loss_func(logits, labels), logits