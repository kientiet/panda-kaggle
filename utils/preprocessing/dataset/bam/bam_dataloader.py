import torch
import math
import random
from torch.utils.data.sampler import RandomSampler, Sampler

class BatchSchedulerSampler(Sampler):
  def __init__(self, dataset, batch_size):
    self.dataset = dataset
    self.batch_size = batch_size
    self.number_of_datasets = len(self.dataset.datasets)
    self.largest_dataset_size = max([dataset.__len__() for dataset in self.dataset.datasets])

  def __len__(self):
    total = math.floor(sum([dataset.__len__() for dataset in self.dataset.datasets]) / self.batch_size)
    return total * self.batch_size

  def __iter__(self):
    samplers_list = []
    sampler_iterators = []
    for dataset_idx in range(self.number_of_datasets):
      cur_dataset = self.dataset.datasets[dataset_idx]
      sampler = RandomSampler(cur_dataset)
      samplers_list.append(sampler)
      cur_sampler_iterator = sampler.__iter__()
      sampler_iterators.append(cur_sampler_iterator)


    push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
    samples_to_grab = self.batch_size
    epoch_samples = self.__len__()

    final_samples_list = []  # this is a list of indexes from the combined dataset
    for _ in range(0, epoch_samples):
      i = random.choices([0, 1], weights = [0.5, 0.5], k = 1)[0]
      cur_batch_sampler = sampler_iterators[i]
      cur_samples = []
      should_keep = True
      for count in range(samples_to_grab):
        try:
          cur_sample_org = cur_batch_sampler.__next__()
          cur_sample = cur_sample_org + push_index_val[i]
          cur_samples.append(cur_sample)
        except StopIteration:
          should_keep = False
          break
          # sampler_iterators[i] = samplers_list[i].__iter__()
          # cur_batch_sampler = sampler_iterators[i]
          # cur_sample_org = cur_batch_sampler.__next__()
          # cur_sample = cur_sample_org + push_index_val[i]
          # cur_samples.append(cur_sample)
      if should_keep:
        final_samples_list.extend(cur_samples)
    return iter(final_samples_list)