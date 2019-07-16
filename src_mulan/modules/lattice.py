from __future__ import absolute_import
from __future__ import unicode_literals
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy

from .latticelstm import LatticeLSTM

class LatticebiLm(nn.Module):
  def __init__(self, config, gaz_size, gaz_embedder):
    super(LatticebiLm, self).__init__()
    self.config = config
    self.lattice_list_forward = nn.ModuleList()
    self.lattice_list_backward = nn.ModuleList()
    self.encoder_forward  = LatticeLSTM(self.config['encoder']['input_dim'],self.config['encoder']['hidden_dim'],
                            self.config['dropout_gaz'], gaz_size,
                            self.config['token_embedder']['gaz_dim'], gaz_embedder, True)

    self.encoder_backward = LatticeLSTM(self.config['encoder']['input_dim'],self.config['encoder']['hidden_dim'],
                            self.config['dropout_gaz'], gaz_size,
                            self.config['token_embedder']['gaz_dim'], gaz_embedder, False)

    for i in range(self.config['encoder']['n_layers']):
        self.lattice_list_forward.append(self.encoder_forward)
        self.lattice_list_backward.append(self.encoder_backward)

  def _stack_forward(self, input_, lattice_inp, gaz_embedder, hidden, forward):


      hidden_list = []
      output_list = []
      cur_input = input_
      if forward:
          for layer_i, lattice in enumerate(self.lattice_list_forward):
              hx_i = hidden[0][layer_i,:,:].unsqueeze(0),hidden[1][layer_i,:,:].unsqueeze(0)
              # print(hx_i[0].size())
              h_output, hidden_ = lattice(cur_input, lattice_inp, hx_i)
              
              output_list.append(h_output)
              hidden_list.append(hidden_)
              cur_input = h_output
      else:
          for layer_i, lattice in enumerate(self.lattice_list_backward):
              hx_i = hidden[0][layer_i,:,:].unsqueeze(0),hidden[1][layer_i,:,:].unsqueeze(0)
              h_output, hidden_ = lattice(cur_input, lattice_inp, hx_i)
              output_list.append(h_output)
              hidden_list.append(hidden_)
              cur_input = h_output

      return cur_input, hidden_list, output_list

  def forward(self, inputs, lattice_inp, gaz_embedder, hidden, forward = True):

    if forward:
        output, hidden_list, output_list = self._stack_forward(inputs, lattice_inp, gaz_embedder, hidden, True)
    else:
        output, hidden_list, output_list = self._stack_forward(inputs, lattice_inp, gaz_embedder, hidden, False)
    # print(hidden_list[0][0].size())
    hidden = (torch.cat([hidden_list[i][0].unsqueeze(0) for i in range(len(hidden_list))], 0), torch.cat([hidden_list[i][1].unsqueeze(0) for i in range(len(hidden_list))], 0))  
    return output, hidden, output_list