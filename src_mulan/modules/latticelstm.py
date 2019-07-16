"""Implementation of batch-normalized LSTM."""
import torch
from torch import nn
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn import functional, init
import numpy as np


class WordLSTMCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(WordLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 3 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        init.orthogonal_(self.weight_ih.data)
        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data.set_(weight_hh_data)
        if self.use_bias:
            init.constant_(self.bias.data, val=0)

    def forward(self, input_, hx):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """
        
        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0).expand(input_.size(0), input_.size(1), *self.bias.size()))
        weight_hh_batch = self.weight_hh.expand(batch_size, *self.weight_hh.size())
        wh_b = torch.add(bias_batch, torch.bmm(h_0.expand(input_.size(1),*h_0.size()).transpose(1,0), weight_hh_batch))
        weight_ih_batch = self.weight_ih.expand(batch_size, *self.weight_ih.size())
        wi = torch.bmm(input_, weight_ih_batch)
        f, i, g = torch.split(wh_b + wi, self.hidden_size, dim=2)
        c_1 = torch.sigmoid(f)*c_0.expand(input_.size(1),*c_0.size()).transpose(1,0) + torch.sigmoid(i)*torch.tanh(g)
        return c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class MultiInputLSTMCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(MultiInputLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 3 * hidden_size))
        self.alpha_weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, hidden_size))
        self.alpha_weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
            self.alpha_bias = nn.Parameter(torch.FloatTensor(hidden_size))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('alpha_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        init.orthogonal_(self.weight_ih.data)
        init.orthogonal_(self.alpha_weight_ih.data)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data.set_(weight_hh_data)

        alpha_weight_hh_data = torch.eye(self.hidden_size)
        alpha_weight_hh_data = alpha_weight_hh_data.repeat(1, 1)
        self.alpha_weight_hh.data.set_(alpha_weight_hh_data)

        # The bias is just set to zero vectors.
        if self.use_bias:
            init.constant_(self.bias.data, val=0)
            init.constant_(self.alpha_bias.data, val=0)

    def forward(self, input_, c_input, hx):
        """
        Args:
            batch = 1
            input_: A (batch, input_size) tensor containing input
                features.
            c_input: A  list with size c_num,each element is the input ct from skip word (batch, hidden_size).
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        batch_size = h_0.size(0)#5
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        i, o, g = torch.split(wh_b + wi, self.hidden_size, dim=1)
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        if c_input[0] == []:
            f = 1 - i
            c_1 = f*c_0 + i*g
            h_1 = o * torch.tanh(c_1)
        else:
            c_num = len(c_input[0])
            c_input_var = torch.stack([torch.cat([c_input[i][j].unsqueeze(0) for j in range(len(c_input[i]))], 0) for i in range(batch_size)], 0)
            # print(c_input_var)

            alpha_mask = torch.ones(c_input_var.size(1),c_input_var.size(0),c_input_var.size(2))

            for b in range(c_input_var.size(0)):
                for n in range(c_input_var.size(1)):
                    if torch.equal(c_input_var[b,n,:],torch.zeros_like(c_input_var[b,n,:])):
                        alpha_mask[n,b,:] *= -1000000
            alpha_bias_batch = (self.alpha_bias.unsqueeze(0).expand(batch_size, *self.alpha_bias.size()))
            alpha_wi = torch.add(alpha_bias_batch, torch.mm(input_, self.alpha_weight_ih))
            alpha_wi = alpha_wi.expand(c_num, *alpha_wi.size())

            alpha_weight_hh_batch = (self.alpha_weight_hh.expand(batch_size, *self.alpha_weight_hh.size()))
            alpha_wh = torch.bmm(c_input_var, alpha_weight_hh_batch)
            alpha = torch.sigmoid(alpha_wi + alpha_wh.transpose(1,0))
            alpha = alpha * alpha_mask.cuda()
            alpha = torch.exp(torch.cat([i.unsqueeze(0), alpha],0))
            alpha_sum = alpha.sum(0)
            alpha = torch.div(alpha, alpha_sum)

            merge_i_c = torch.cat([g.unsqueeze(0), c_input_var.transpose(1,0)],0)
            c_1 = merge_i_c * alpha
            c_1 = c_1.sum(0)
            h_1 = o * torch.tanh(c_1)

        
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class LatticeLSTM(nn.Module):

    """A module that runs multiple steps of LSTM."""
    def __init__(self, input_dim, hidden_dim, word_drop, word_alphabet_size, word_emb_dim, gaz_embedder, left2right=True):
        super(LatticeLSTM, self).__init__()
        skip_direction = "forward" if left2right else "backward"
        print("build LatticeLSTM... ", skip_direction, " gaz drop:", word_drop)
        self.hidden_dim = hidden_dim    
        self.word_emb = gaz_embedder 
        self.word_dropout = nn.Dropout(word_drop)
        self.rnn = MultiInputLSTMCell(input_dim, hidden_dim)
        self.word_rnn = WordLSTMCell(word_emb_dim, hidden_dim)
        self.left2right = left2right
        self.rnn = self.rnn.cuda()
        self.word_emb = self.word_emb.cuda()
        self.word_dropout = self.word_dropout.cuda()
        self.word_rnn = self.word_rnn.cuda()
        print("build LatticeLSTM End... ")

    def forward(self, input_,  lattice_input, hidden):

        """
            input_: variable (batch, seq_len), batch = 1
            skip_input_list: [skip_input, volatile_flag]
            skip_input: three dimension list, with length is seq_len. Each element is a list of matched word id and its length. 
                        example: [[], [[25,13],[2,3]]] 25/13 is word id, 2,3 is word length . 
        """
        
        seq_len = input_.size(0)
        batch_size = input_.size(1)

        hidden_out = []
        memory_out = []
        (hx,cx)= hidden
        
        hx = hx.squeeze(0).cuda()
        cx = cx.squeeze(0).cuda()
        id_list = list(range(seq_len))
        input_c_list = init_list_of_objects(seq_len, batch_size)
        for t in id_list:

            (hx,cx) = self.rnn(input_[t], input_c_list[t], (hx,cx))

            hidden_out.append(hx.unsqueeze(0))
            memory_out.append(cx.unsqueeze(0))

            word_embs =  self.word_emb(autograd.Variable(torch.LongTensor(lattice_input[t])))

            word_embs = self.word_dropout(word_embs)
            ct = self.word_rnn(word_embs, (hx,cx))

            for i in range(batch_size):
                    if t+1 < seq_len:
                        if lattice_input[t][i][0] == 0:
                            input_c_list[t+1][i].append(torch.zeros_like(ct[i,0,:]))
                        else:
                            input_c_list[t+1][i].append(ct[i,0,:])
                    if t+2 < seq_len:
                        if lattice_input[t][i][1] == 0:
                            input_c_list[t+2][i].append(torch.zeros_like(ct[i,1,:]))
                        else:
                            input_c_list[t+2][i].append(ct[i,1,:])
                    if t+3 < seq_len:
                        if lattice_input[t][i][2] == 0:
                            input_c_list[t+3][i].append(torch.zeros_like(ct[i,2,:]))
                        else:
                            input_c_list[t+3][i].append(ct[i,2,:])

        output_hidden, output_memory = torch.cat(hidden_out, 0), torch.cat(memory_out, 0)
        return output_hidden, (output_hidden[-1,:,:], output_memory[-1,:,:])


def init_list_of_objects(size, bsz):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( [list() for i in range(bsz)] )
    return list_of_objects


