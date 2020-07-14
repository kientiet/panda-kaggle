import os
import torch
import pytorch_lightning as pl
from trainer.supervised.trainer import SupervisedTrainerSkeleton

from models.resnet import ResNetModel
from models.project_layer import ProjectLayer

config_dir = os.path.join(os.getcwd(), "config", "resnet.yaml")

class BaselineTrainer(SupervisedTrainerSkeleton):
  def __init__(self, trainset = None, valset = None):
    super().__init__(yaml_file = config_dir, trainset = trainset, valset = valset)
    self.init_model()


  def init_model(self):
    self.encoder = ResNetModel(self.stream["backbone"], pretrained = self.stream["pretrained"])
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

    if self.stream["loss_func"] == "bceloss":
      logits = torch.sigmoid(logits)

      return self.loss_func(logits.to(torch.double), labels), logits
    else:
      return self.loss_func(logits / self.temperature, labels), logits