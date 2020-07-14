import os
import torch
import pytorch_lightning as pl
from trainer.supervised.trainer import SupervisedTrainerSkeleton

# Models
from models.resnet import ResNetModel
from models.transformer_project import TransformerProjectLayer

config_dir = os.path.join(os.getcwd(), "config", "attention.yaml")

class TransformersTrainer(SupervisedTrainerSkeleton):
  def __init__(self):
    super(SupervisedTrainerSkeleton, self).__init__(config_dir)
    self.init_encoder()


  def init_encoder(self):
    self.encoder = ResNetModel(self.stream["backbone"], pretrained = self.stream["pretrained"])

    self.transformers = TransformerProjectLayer(num_channel = self.encoder.num_channel,
                                    num_classes = self.stream["num_classes"],
                                    d_model = self.stream["d_model"],
                                    n_head = self.stream["n_head"],
                                    dim_forward = self.stream["dim_forward"],
                                    dropout_rate = self.stream["attention_dropout"],
                                    num_position = self.stream["image_batch"],
                                    num_layers = self.stream["num_layers"],
                                    batch_size = self.stream["batch_size"]
                                    )

    self.configure_optimizers(self.stream)


  def forward(self, images, labels):
    batch_size, image_batch, channels, height, width = images.shape
    images = images.reshape(-1, channels, height, width)

    # ResNet + Transformation for images
    logits = self.encoder(images)

    # Get attention scores
    scores = self.transformers(logits)

    return self.loss_func(scores.to(torch.double), labels), scores
