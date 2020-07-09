import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Transformers(nn.Module):
  def __init__(self,
              num_position,
              d_model,
              n_head,
              num_layers,
              dim_feedforward,
              dropout_rate = 0.1):

    super().__init__()
    ## Get the positional encoding by channel as the features
    self.position_encoding = PositionEmbedding(d_model, num_position)

    ## Get a Transformer block
    self_attention_layer = TransformerLayer(d_model, n_head, dim_feedforward, dropout_rate)

    ## Stack blocks
    self.encoder = TransformersBlock(self_attention_layer, num_layers)

    ## Store hyperparameters
    self.dim_feedforward = dim_feedforward
    self.init_weight()

  def init_weight(self):
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)


  def forward(self, inputs):
    # Get the positional encoding
    position_encoding = self.position_encoding(inputs)
    outputs = self.encoder(inputs, position_encoding)
    return outputs


class TransformersBlock(nn.Module):
  def __init__(self, layer, num_layers):
    super().__init__()

    self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
    self.num_layers = num_layers

  def forward(self, inputs, pos):
    outputs = inputs
    for layer in self.layers:
      outputs = layer(outputs, pos)

    return outputs


class TransformerLayer(nn.Module):
  def __init__(self,
              d_model,
              n_head,
              dim_forward = 2048,
              dropout_rate = 0.1
              ):

    super().__init__()
    self.self_attention = nn.MultiheadAttention(d_model, n_head, dropout_rate)

    self.linear1 = nn.Linear(d_model, dim_forward)
    self.linear2 = nn.Linear(dim_forward, d_model)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)

    self.dropout = nn.Dropout(dropout_rate)
    self.activation = nn.ReLU()

  def forward(self, inputs, pos):
    # Position encoding + inputs
    q = k = inputs + pos

    scores = self.self_attention(q, k, value = inputs)[0]
    inputs = inputs + self.dropout(scores)
    inputs = self.norm1(inputs)

    features = self.linear2(self.dropout(self.activation(self.linear1(inputs))))
    features = features + self.dropout(inputs)
    features = self.norm2(features)

    return features


class PositionEmbedding(nn.Module):
  def __init__(self, embed_dim, num_position):
    super().__init__()
    self.embed = nn.Embedding(num_position, embed_dim)

    self.init_weight()


  def init_weight(self):
    nn.init.uniform_(self.embed.weight)


  def forward(self, inputs):
    sequence = inputs.shape[1]
    i = torch.arange(sequence, device = inputs.device)
    emb = self.embed(i)

    pos = emb.repeat(inputs.shape[0], 1, 1)

    return pos