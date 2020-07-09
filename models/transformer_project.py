import torch
import torch.nn as nn

from models.transformers import Transformers
from activation.mish import Mish

class TransformerProjectLayer(nn.Module):
  def __init__(self,
              num_channel,
              num_classes,
              is_pooling = True,
              **kwags
              ):

    super().__init__()
    self.num_classes = num_classes
    self.downsample = nn.Conv2d(num_channel, kwags["d_model"], kernel_size = 1, bias = False)
    self.activation = Mish()
    self.bn = nn.BatchNorm2d(kwags["d_model"])
    self.kwags = kwags

    self.transformers = Transformers(num_position = kwags["num_position"],
                                    num_layers = kwags["num_layers"],
                                    d_model = kwags["d_model"],
                                    n_head = kwags["n_head"],
                                    dim_feedforward = kwags["dim_forward"],
                                    dropout_rate = kwags["dropout_rate"])
    # Project head
    self.head = nn.Sequential(nn.Linear(kwags["d_model"], kwags["d_model"]),
                              Mish(),
                              nn.BatchNorm1d(kwags["num_position"]),
                              nn.Dropout(0.5),
                              )

    self.learned_mean = nn.Sequential(nn.Linear(kwags["d_model"], 1),
                                      nn.Flatten(),
                                      nn.Softmax(dim = -1))
    self.prob_layer = nn.Linear(kwags["d_model"], num_classes)

  def forward(self, inputs):
    # Batch x image_batch x dim
    # breakpoint()
    batch, channels, height, width = inputs.shape
    inputs = self.downsample(inputs)
    inputs = self.bn(self.activation(inputs))
    # Get mean of image
    inputs = inputs.reshape(batch // self.kwags["num_position"], self.kwags["num_position"], self.kwags["d_model"], -1)
    inputs = torch.mean(inputs, dim = -1)

    features = self.transformers(inputs)
    attention = self.head(features)

    prob = self.prob_layer(attention)
    scores = self.learned_mean(attention)

    outputs = torch.mul(scores[:, :, None].repeat(1, 1, self.num_classes), prob)
    outputs = torch.sigmoid(torch.sum(outputs, dim = 1))
    return outputs