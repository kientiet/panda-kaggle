import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.preprocessing.split.bam.student_train_test_split import load_for_bam_student
from utils.preprocessing.dataset.bam.bam_dataloader import BatchSchedulerSampler
from trainer.distill.trainer import DistillTrainerSkeleton

# Model Architecture
from models.custom_resnet_wrapper import CustomResNetModel
from models.custom_efficientnet import CustomEfficientNet
from models.project_layer import ProjectLayer

# Teacher Architecture
from trainer.supervised.baseline import BaselineTrainer

config_dir = os.path.join(os.getcwd(), "config", "student_multi.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"

batch_norm_order = {
  "karolinska": 0,
  "radboud": 1
}

class StudentMultiTrainer(DistillTrainerSkeleton):
  def __init__(self, train_table, val_table):
    super(StudentMultiTrainer, self).__init__(config_dir)
    ## Init for Ban Student
    self.init_model(self.stream, train_table, val_table)

  def init_model(self, stream, train_table, val_table):
    # Reload dataset
    self.trainset, self.valset = load_for_bam_student(train_table, val_table,
                                                      transform = [self.stream["train_aug"], self.stream["test_aug"]],
                                                      data_dir = self.stream["data_dir"],
                                                      loss_type = self.stream["loss_type"],
                                                      label_type = self.stream["label_type"],
                                                      sample = self.stream["sample"]
                                                      )
    print("\n\n>> The transformation of student is")
    print(self.trainset.datasets[0].transform)
    self.trainloader = torch.utils.data.DataLoader(dataset = self.trainset,
                                        sampler = BatchSchedulerSampler(dataset = self.trainset, batch_size = self.stream["batch_size"]),
                                        batch_size = self.stream["batch_size"], shuffle = False, num_workers = 4)

    # self.trainloader = torch.utils.data.DataLoader(dataset = self.trainset, batch_size = self.stream["batch_size"], shuffle = True, num_workers = 4)

    self.valloader = torch.utils.data.DataLoader(dataset = self.valset,
                                        sampler = BatchSchedulerSampler(dataset = self.valset, batch_size = self.stream["val_batch_size"]),
                                        batch_size = self.stream["val_batch_size"], shuffle = False, num_workers = 4)
    # self.valloader = torch.utils.data.DataLoader(self.valset, batch_size = self.stream["batch_size"], shuffle = False, num_workers = 4)

    # Update the dataloader
    self.train_dataloader()
    self.val_dataloader()

    self.config_model()


  def config_model(self):
    # Init student encoder and project_layer
    # self.encoder = CustomResNetModel(self.stream["backbone"], pretrained = self.stream["pretrained"], \
    #   stochastic_depth_prob = self.stream["stochastic_depth_prob"], split_batch_norm = self.stream["split_batch_norm"])

    self.encoder = CustomEfficientNet.from_pretrained(self.stream["backbone"], num_classes = self.stream["num_classes"])
    # self.project_layer = nn.ModuleDict({
    #   "karolinska": ProjectLayer(self.encoder.num_channel, self.stream["num_classes"]),
    #   "radboud": ProjectLayer(self.encoder.num_channel, self.stream["num_classes"])
    # })

    # self.project_layer = ProjectLayer(self.encoder.num_channel, self.stream["num_classes"])
    # Optimzer and scheduler
    self.configure_optimizers(self.stream)

    # Init the loss function for student
    self.loss_func = self.loss_func(self.stream["anneal_type"], coeff = self.stream["coeff"], temperature = self.stream["temperature"], \
      total_step = self.get_total_step(), label_type = self.stream["label_type"])

  def get_grad_parameters(self):
    params = []
    if self.stream["layer_wise"]:
      print(">> layer-wise learning rate")
      if isinstance(self.encoder, CustomResNetModel):
        # Get the maximum layer
        print(">> In resnet")
        max_layer = -1
        for name, p in self.encoder.named_parameters():
          if p.requires_grad:
            name = name.split(".")
            if "layer" not in name[1]: continue
            max_layer = max(max_layer, int(name[1].split("layer")[1]))

        # Append layer with their learning rate
        current_layer, current_params = 0, []
        max_lr = self.stream["max_lr"]
        for name, p in self.encoder.named_parameters():
          name = name.split(".")[1]
          if "layer" not in name[1]: layer = 0
          else:
            layer = name.split("layer")[1]

          if layer != current_layer:
            params.append({"params": current_params, "lr": max_lr * (self.stream["decay_layer"] ** (max_layer - current_layer))})
            current_layer = layer
            current_params = [p]
          else:
            current_params.append(p)

        params.append({"params": current_params, "lr": max_lr / (self.stream["decay_layer"] ** (max_layer - current_layer))})

        if isinstance(self.project_layer, dict):
          # Learning rate is the same for the linear classifier
          for value in self.project_layer.values():
            params.append({"params": value.parameters()})
        else:
          params.append({"params": self.project_layer.parameters()})
      else:
        params = self.encoder.layer_wise_parameters(self.stream["max_lr"], self.stream["decay_layer"])
    else:
      print(">> Same learning rate")
      for name, param in self.encoder.named_parameters():
        if param.requires_grad:
          params.append(param)
      # Learning rate is the same for the linear classifier
      for value in self.project_layer.values():
        params = params + list(value.parameters())

    return params


  def forward(self, images, hard_labels, soft_labels, data_provider, task_weights, running_mode):
    # Prepare the batch
    # batch_size, image_batch, channels, height, width = images.shape

    # images = images.reshape(-1, channels, height, width)

    ## Forward pass for student
    batch_norm_index = 0
    if self.stream["split_batch_norm"]:
      ## ! In order to use split_batch_norm, the batch should be selected
      ## ! from the same data_provider
      assert len(np.unique(data_provider)) == 1
      batch_norm_index = batch_norm_order[data_provider[0]]

    logits = self.encoder(images, batch_norm_index)
    # shape = logits.shape
    # concatenate the output for tiles into a single map
    # logits = logits.view(-1, image_batch, shape[1], shape[2], shape[3]).permute(0, 2, 1, 3, 4).contiguous() \
    #   .view(-1, shape[1], shape[2] * image_batch, shape[3])

    '''
      TODO: Swap to check the confident
    '''
    if hasattr(self, "project_layer"):
      if isinstance(self.project_layer, dict):
        # Get the correct layers
        data_provider = np.array(data_provider)
        final_logits, total_index = [], []

        for provider in np.unique(data_provider):
          index = np.where(data_provider == provider)[0]
          if len(index) > 1:
            total_index = np.append(total_index, index)
            temp = self.project_layer[provider](logits[index])

            final_logits.append(temp)

        final_logits = torch.cat(final_logits)
        total_index = torch.from_numpy(total_index.astype(int)).to(device = final_logits.device)
        task_weights = torch.gather(task_weights, 0, total_index)
        hard_labels = torch.gather(hard_labels, 0, total_index)
        soft_labels = soft_labels[total_index]
      else:
        final_logits = self.project_layer(logits)

    if running_mode == "training":
      # Get loss
      loss, hard_loss, soft_loss, coeff = self.loss_func(logits, hard_labels,
                                                        soft_labels = soft_labels,
                                                        task_weights = task_weights,
                                                        running_mode = running_mode
                                                        )

      return loss, hard_loss, soft_loss, coeff
    else:
      loss = self.loss_func(logits, hard_labels, running_mode = running_mode)
      return loss, logits


  def training_step(self, train_batch, batch_idx):
    images, hard_labels, soft_labels, data_provider, task_weight = train_batch
    loss, hard_loss, soft_loss, coeff = self.forward(images, hard_labels, soft_labels, data_provider, task_weight, "training")

    logs = {"learning_rate": self.optimizer.param_groups[-1]["lr"],
            "train_loss": loss.item(),
            "hard_loss": hard_loss.item(),
            "soft_loss": soft_loss.item(),
            "coeff": coeff
            }

    return {"loss": loss, "log": logs}


  def validation_step(self, val_batch, val_idx):
    images, hard_labels, soft_labels, data_provider, task_weight = val_batch
    val_loss, logits  = self.forward(images, hard_labels, soft_labels, data_provider, task_weight, "validation")

    preds = torch.max(F.softmax(logits, dim = -1), dim = -1)[1]
    logs = {"val_loss": val_loss,
            "y_true": hard_labels.cpu().numpy(),
            "data_provider": np.array(data_provider),
            "y_pred": preds.cpu().numpy()}

    return logs


  def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
    self.optimizer.step()
    self.optimizer.zero_grad()

    self.scheduler.step()

    if hasattr(self, "project_layer"):
      if isinstance(self.project_layer, dict):
        for data_provider, project_layer in self.project_layer.items():
          weight = project_layer.head[1].weight.view(-1)
          self.logger.experiment.add_histogram("{}_project_layer_1".format(data_provider), weight, self.global_step)

          weight = project_layer.head[5].weight.view(-1)
          self.logger.experiment.add_histogram("{}_project_layer_5".format(data_provider), weight, self.global_step)
      else:
          weight = self.project_layer.head[1].weight.view(-1)
          self.logger.experiment.add_histogram("project_layer_1", weight, self.global_step)

          weight = self.project_layer.head[5].weight.view(-1)
          self.logger.experiment.add_histogram("project_layer_5", weight, self.global_step)
