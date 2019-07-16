
import torch
import torch.nn as nn
from torch.autograd import Variable

class ConvTokenEmbedder(nn.Module):
  def __init__(self, word_emb_layer):
    super(ConvTokenEmbedder, self).__init__()

    self.word_emb_layer = word_emb_layer

  def forward(self, word_inp):
    embs = []
    if self.word_emb_layer is not None:
      batch_size, seq_len = word_inp.size(0), word_inp.size(1)
      word_emb = self.word_emb_layer(Variable(word_inp).cuda())
      embs.append(word_emb)
      
    token_embedding = torch.cat(embs, dim=2)

    return token_embedding
