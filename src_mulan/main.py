import os
import errno
import sys
import codecs
import argparse
import time
import random
import logging
import json
import collections
import os.path as op

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from util.gazetteer import Gazetteer
from modules.token_embedder import ConvTokenEmbedder
from modules.embedding_layer import EmbeddingLayer
from modules.classify_layer import SoftmaxLayer
from modules.lattice import LatticebiLm


def dict2namedtuple(dic):
  return collections.namedtuple('Namespace', dic.keys())(**dic)

def read_test_corpus(word_lexicon, path):
  data = []
  with codecs.open(path, 'r', encoding='utf-8') as fin:
    for line in fin:
      data.append('<bos>')
      for token in line.strip().split():
        data.append(token)

      data.append('<eos>')

  data_id = torch.LongTensor(len(data))
  token = 0
  for char in data:
    if char in word_lexicon:
      data_id[token] = word_lexicon[char]
    else:
      data_id[token] = word_lexicon['<oov>']
    token += 1

  data_id_back = data_id.tolist()
  data_id_back.reverse()
  data_id_back = torch.LongTensor(data_id_back)

  return data_id, data_id_back

def read_corpus(word_lexicon, path, test_path):

  data = []
  data_test = []
  with codecs.open(path, 'r', encoding='utf-8') as fin:
    for line in fin:
      data.append('<bos>')
      for token in line.strip().split():
        data.append(token)
        if token not in word_lexicon:
          word_lexicon[token] = len(word_lexicon)
      data.append('<eos>')

  with codecs.open(test_path, 'r', encoding='utf-8') as fin:
    for line in fin:
      data_test.append('<bos>')
      for token in line.strip().split():
        data_test.append(token)
        if token not in word_lexicon:
          word_lexicon[token] = len(word_lexicon)
      data_test.append('<eos>')

  data_id = torch.LongTensor(len(data))
  token = 0
  for char in data:
    data_id[token] = word_lexicon[char]
    token += 1
  data_id_back = data_id.tolist()
  data_id_back.reverse()
  data_id_back = torch.LongTensor(data_id_back)  


  data_test_id = torch.LongTensor(len(data_test))
  token = 0
  for char in data_test:
    data_test_id[token] = word_lexicon[char]
    token += 1
  data_test_id_back = data_test_id.tolist()
  data_test_id_back.reverse()
  data_test_id_back = torch.LongTensor(data_test_id_back)  

  return data_id, data_id_back, data_test_id, data_test_id_back, word_lexicon

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data

def batchify_gaz(data, bsz):
    nbatch = len(data) // bsz
    data = data[0 : nbatch * bsz]
    data = np.reshape(np.array(data), (bsz, nbatch, -1)).transpose(1,0,2)
    return data

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                if (embedd_dim + 1 != len(tokens)):
                    continue
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim

def load_embedding(word_embedding, word_lexicon):
    embedd_dict = dict()
    embedd_dict, embedd_dim = load_pretrain_emb(word_embedding)
    scale = np.sqrt(3.0 / embedd_dim)
    perfect_match = 0
    not_match = 0
    words = []
    vals = []

    word_lexion_inx2word = {v:k for k,v in word_lexicon.items() }
    for idx in range(len(word_lexion_inx2word)):
      words.append(word_lexion_inx2word[idx])
      if word_lexion_inx2word[idx] in embedd_dict:
        vals += [embedd_dict[word_lexion_inx2word[idx]]]
        perfect_match += 1  
      else:
        vals += [np.random.uniform(-scale, scale, [1, embedd_dim])]
        not_match += 1  

    # for word in word_lexicon:
    #     words.append(word)
    #     if word in embedd_dict:
    #         vals += [embedd_dict[word]]
    #         perfect_match += 1  
    #     else:
    #         vals += [np.random.uniform(-scale, scale, [1, embedd_dim])]
    #         not_match += 1
    logging.info("prefect match:%s, oov:%s, oov%%:%s"%(perfect_match, not_match, (not_match+0.)/(perfect_match + not_match)))
    
    return words, np.asarray(vals).reshape(len(words), -1)

def eval_model(model, batch_size, bptt, test,  test_back, test_lattice, test_lattice_back):
  model.eval()
  total_loss = 0
  total_loss_back = 0
  hidden = model.init_hidden(batch_size)
  hidden_back = model.init_hidden(batch_size)
  cnt = 0
  for batch, i in enumerate(range(1, test.size(0) - 1, bptt)):
    cnt += 1
    data, data_back, lattice, lattice_back, targets_forw, targets_back = get_batch(test, test_back, test_lattice, test_lattice_back, i, bptt)
    hidden = repackage_hidden(hidden)
    hidden_back = repackage_hidden(hidden_back)
    loss, hidden, loss_back, hidden_back = model.forward(data, data_back, lattice, lattice_back, targets_forw, targets_back, hidden, hidden_back)
    # hidden = hidden[0].unsqueeze(0), hidden[1].unsqueeze(0)
    # hidden_back = hidden_back[0].unsqueeze(0), hidden_back[1].unsqueeze(0)
    total_loss += loss.data
    total_loss_back += loss_back.data

  test_loss = total_loss / cnt
  test_loss_back = total_loss_back / cnt
  test_ppl = np.exp(test_loss)
  test_ppl_back = np.exp(test_loss_back)
  model.train()

  return test_ppl, test_ppl_back

def test():
  cmd = argparse.ArgumentParser('The testing components of')
  cmd.add_argument("--model", required=True, help="path to save model")
  cmd.add_argument("--batch_size", type=int, default=64, help='the batch size.')
  cmd.add_argument('--bptt', type=int, default=40, help='maximum sentence length.')
  cmd.add_argument('--test_path', required=True, help='The path to the test file.')
  cmd.add_argument("--gaz_file", default='../data/emb/lm_mincount3_gaz200_len234_with_zeros', help="The path to gaz vectors.")

  args = cmd.parse_args(sys.argv[2:])
  args2 = dict2namedtuple(json.load(codecs.open(os.path.join(args.model, 'config.json'), 'r', encoding='utf-8')))

  with open(args2.config_path, 'r') as fin:
    config = json.load(fin)

  word_lexicon = {}
  with codecs.open(os.path.join(args.model, 'word.dic'), 'r', encoding='utf-8') as fpi:
    for line in fpi:
      tokens = line.strip().split('\t')
      if len(tokens) == 1:
        tokens.insert(0, '\u3000')
      token, i = tokens
      word_lexicon[token] = int(i)

  if config['token_embedder']['word_dim'] > 0:
    word_emb_layer = EmbeddingLayer(config['token_embedder']['word_dim'], word_lexicon, fix_emb=False, embs=None)
    # logging.info('word embedding size: ' + str(len(word_emb_layer.word2id)))
  else:
    word_emb_layer = None

  gaz_lexicon = {}
  with codecs.open(os.path.join(args.model, 'gaz.dic'), 'r', encoding='utf-8') as fpi:
    for line in fpi:
      tokens = line.strip().split('\t')
      if len(tokens) == 1:
        tokens.insert(0, '\u3000')
      token, i = tokens
      gaz_lexicon[token] = int(i)

  logging.info('read gaz lexicon...')
  gaz = Gazetteer(False)
  gaz.build_gaz_file(os.path.join(args.model, 'gaz.dic'))

  logging.info('build gaz data...')
  test_lattice_Ids, test_lattice_Ids_back = gaz.build_test_gaz_lexicon(args.test_path, gaz_lexicon)

  test_lattice_Ids_back.reverse()

  gaz = None
  test_lattice_Ids = batchify_gaz(test_lattice_Ids, args.batch_size)
  test_lattice_Ids_back = batchify_gaz(test_lattice_Ids_back, args.batch_size)

  gaz_emb_layer = EmbeddingLayer(config['token_embedder']['gaz_dim'], gaz_lexicon, fix_emb=True, normalize=False, embs=None)

  model = Model(config, word_emb_layer, gaz_emb_layer, len(word_lexicon), len(gaz_lexicon))

  model.cuda()
  model.load_model(args.model)

  test_data_id, test_data_id_back = read_test_corpus(word_lexicon, args.test_path)
  test_data_id = batchify(test_data_id, args.batch_size)
  test_data_id_back = batchify(test_data_id_back, args.batch_size)
  
  test_ppl, test_ppl_back = eval_model(model, args.batch_size, args.bptt, test_data_id, test_data_id_back, test_lattice_Ids, test_lattice_Ids_back)

  logging.info('Test Result : ppl {:8.2f} | ppl_back{:8.2f} '.format(test_ppl, test_ppl_back))

def train():
  cmd = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
  cmd.add_argument('--seed', default=1, type=int, help='The random seed.')
  cmd.add_argument('--train_path', required=True, help='The path to the training file.')
  cmd.add_argument('--dev_path', required=True, help='The path to the test file.')
  cmd.add_argument('--config_path', required=True, help='the path to the config file.')
  cmd.add_argument("--word_embedding", default='../data/emb/your_char_emb_file', help="The path to word vectors.")
  cmd.add_argument("--gaz_file", default='../data/emb/your_gaz_emb_file', help="The path to gaz vectors.")
  cmd.add_argument('--optimizer', default='adam', choices=['sgd', 'adam', 'adagrad'], help='optimizer')
  cmd.add_argument("--lr", type=float, default=0.0001, help='the learning rate.')
  cmd.add_argument("--lr_decay", type=float, default=0.8, help='the learning rate decay.')
  cmd.add_argument("--model", required=True, help="path to save model")
  cmd.add_argument("--batch_size", type=int, default=64, help='the batch size.')
  cmd.add_argument("--max_epoch", type=int, default=15, help='the maximum number of iteration.')
  cmd.add_argument("--clip_grad", type=float, default=5, help='the tense of clipped grad.')
  cmd.add_argument('--bptt', type=int, default=40, help='maximum sentence length.')
  cmd.add_argument('--eval_steps', type=int, default=5000, help='report every xx batches.')

  opt = cmd.parse_args(sys.argv[2:])

  with open(opt.config_path, 'r') as fin:
    config = json.load(fin)

  # Dump configurations
  print(opt)
  print(config)

  torch.manual_seed(opt.seed)
  random.seed(opt.seed)
  torch.cuda.manual_seed(opt.seed)
  np.random.seed(opt.seed)


  #build gazs

  logging.info('read gaz lexicon...')
  gaz = Gazetteer(False)
  gaz.build_gaz_file(opt.gaz_file)
  gaz_lexicon = {}
  for special_word in ['<zeros>', '<oov>', '<pad>']:
    if special_word not in gaz_lexicon:
      gaz_lexicon[special_word] = len(gaz_lexicon)

  logging.info('build gaz data...')
  train_lattice_Ids, train_lattice_Ids_back, dev_lattice_Ids, dev_lattice_Ids_back, gaz_lexicon = gaz.build_gaz_lexicon(opt.train_path, opt.dev_path, gaz_lexicon)

  train_lattice_Ids_back.reverse()
  dev_lattice_Ids_back.reverse()

  gaz = None
  train_lattice_Ids = batchify_gaz(train_lattice_Ids, opt.batch_size)
  train_lattice_Ids_back = batchify_gaz(train_lattice_Ids_back, opt.batch_size)
  dev_lattice_Ids = batchify_gaz(dev_lattice_Ids, opt.batch_size)
  dev_lattice_Ids_back = batchify_gaz(dev_lattice_Ids_back, opt.batch_size)
  
  gaz_embs = load_embedding(opt.gaz_file, gaz_lexicon)
  gaz_emb_layer = EmbeddingLayer(config['token_embedder']['gaz_dim'], gaz_lexicon, gaz_embs, fix_emb=True, normalize=False)
  logging.info('build gaz data end...')

  #build words 
  word_lexicon = {}
  for special_word in ['<oov>', '<bos>', '<eos>',  '<pad>']:
    if special_word not in word_lexicon:
      word_lexicon[special_word] = len(word_lexicon)

  train_data_id, train_data_id_back, dev_data_id, dev_data_id_back, word_lexicon = read_corpus(word_lexicon, opt.train_path, opt.dev_path)
  train_data_id = batchify(train_data_id, opt.batch_size)
  train_data_id_back = batchify(train_data_id_back, opt.batch_size)
  dev_data_id = batchify(dev_data_id, opt.batch_size)
  dev_data_id_back = batchify(dev_data_id_back, opt.batch_size)
  
  if opt.word_embedding:
    embs = load_embedding(opt.word_embedding, word_lexicon)
  else:
    embs = None
  word_emb_layer = EmbeddingLayer(config['token_embedder']['word_dim'], word_lexicon, embs, fix_emb=False)

  model = Model(config, word_emb_layer, gaz_emb_layer, len(word_lexicon), len(gaz_lexicon))
  model = model.cuda()

  # if opt.model is not None and op.exists(opt.model):
  #   model.load_model(opt.model)
  #   logging.info('continue training...')

  need_grad = lambda x: x.requires_grad
  if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(filter(need_grad, model.parameters()), lr=opt.lr)
  elif opt.optimizer.lower() == 'sgd':
    optimizer = optim.SGD(filter(need_grad, model.parameters()), lr=opt.lr)
  elif opt.optimizer.lower() == 'adagrad':
    optimizer = optim.Adagrad(filter(need_grad, model.parameters()), lr=opt.lr)
  else:
    raise ValueError('Unknown optimizer {}'.format(opt.optimizer.lower()))

  try:
    os.makedirs(opt.model)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise

  with codecs.open(os.path.join(opt.model, 'word.dic'), 'w', encoding='utf-8') as fpo:
    for w, i in word_lexicon.items():
      print('{0}\t{1}'.format(w, i), file=fpo)

  with codecs.open(os.path.join(opt.model, 'gaz.dic'), 'w', encoding='utf-8') as fpo:
    for w, i in gaz_lexicon.items():
      print('{0}\t{1}'.format(w, i), file=fpo)

  json.dump(vars(opt), codecs.open(os.path.join(opt.model, 'config.json'), 'w', encoding='utf-8'))

  best_dev = 1e+8

  for epoch in range(opt.max_epoch):
    best_test = train_model_with_gaz(epoch, opt, model, optimizer, train_data_id, train_data_id_back, train_lattice_Ids, train_lattice_Ids_back, dev_data_id, dev_data_id_back, dev_lattice_Ids, dev_lattice_Ids_back, best_dev)
    if opt.lr_decay > 0:
      optimizer.param_groups[0]['lr'] *= opt.lr_decay




def get_batch(char_source, char_source_back, lattice_source, lattice_back_source, i, bptt):
    seq_len = min(bptt, len(char_source) - 1 - i)
    data = Variable(char_source[i:i+seq_len])
    lattice = lattice_source[i:i+seq_len]
    target_forw = Variable(char_source[i+1:i+1+seq_len].view(-1))

    data_back = Variable(char_source_back[i:i+seq_len])
    lattice_back = lattice_back_source[i:i+seq_len]
    target_back = Variable(char_source_back[i+1:i+1+seq_len].view(-1))
    return data.cuda(), data_back.cuda(), lattice, lattice_back, target_forw.cuda(), target_back.cuda()

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == torch.Tensor:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def train_model_with_gaz(epoch, opt, model, optimizer, train, train_back, train_lattice, train_lattice_back, dev, dev_back, dev_lattice, dev_lattice_back, best_dev):
  model.train()
  total_loss = 0
  total_loss_back = 0
  hidden = model.init_hidden(opt.batch_size)
  hidden_back = model.init_hidden(opt.batch_size)
  temp_start = time.time()
  for batch, i in enumerate(range(1, train.size(0) - 1, opt.bptt)):
    data, data_back, lattice, lattice_back, targets_forw, targets_back = get_batch(train, train_back, train_lattice, train_lattice_back, i, opt.bptt)

    hidden = repackage_hidden(hidden)
    hidden_back = repackage_hidden(hidden_back)
    loss, hidden, loss_back, hidden_back = model.forward(data, data_back, lattice, lattice_back, targets_forw, targets_back, hidden, hidden_back)
    total_loss += loss.data
    total_loss_back += loss_back.data
    l = ( loss + loss_back ) 
    l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip_grad)
    optimizer.step()

    if batch % opt.eval_steps == 0 and batch > 0:
      train_loss = total_loss / opt.eval_steps
      train_loss_back = total_loss_back / opt.eval_steps
      train_ppl = np.exp(train_loss)
      train_ppl_back = np.exp(train_loss_back)

      total_loss = 0
      total_loss_back = 0

      logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:5.8f} | ppl {:8.2f} | ppl_back {:8.2f} '.format(
        epoch, batch, len(train) // opt.bptt, optimizer.param_groups[0]['lr'], train_ppl, train_ppl_back))  
  dev_ppl, dev_ppl_back = eval_model(model, opt.batch_size, opt.bptt, dev,  dev_back, dev_lattice, dev_lattice_back)
  logging.info("---------dev_ppl={:.6f} dev_ppl_back={:.6f}---------".format(dev_ppl, dev_ppl_back))
  if dev_ppl < best_dev:
    best_dev = dev_ppl
    logging.info("---------New record achieved on dev dataset!---------")
    model.save_model(opt.model)  

  return best_dev

class Model(nn.Module):
  def __init__(self, config, word_emb_layer, gaz_emb_layer, n_class, gaz_size):
    super(Model, self).__init__() 
    self.config = config
    self.token_embedder = ConvTokenEmbedder(word_emb_layer)
    self.gaz_embedder = ConvTokenEmbedder(gaz_emb_layer)
    self.encoder = LatticebiLm(config, gaz_size, self.gaz_embedder)
    self.output_dim = config['encoder']['hidden_dim']
    self.classify_layer = SoftmaxLayer(self.output_dim, n_class)

  def forward(self, word_inp, word_inp_back, input_lattice, input_lattice_back, targets_forw, targets_back, hidden, hidden_back):

    token_embedding = self.token_embedder(word_inp)
    token_embedding = F.dropout(token_embedding, self.config['dropout'], self.training)
    output, hidden_list, output_list = self.encoder(token_embedding, input_lattice, self.gaz_embedder, hidden, True)
  
    token_embedding_back = self.token_embedder(word_inp_back)
    token_embedding_back = F.dropout(token_embedding_back, self.config['dropout'], self.training)
    output_back, hidden_list_back, output_list_back = self.encoder(token_embedding_back, input_lattice_back, self.gaz_embedder, hidden_back, False)

    return self.classify_layer(output, targets_forw), hidden_list, self.classify_layer(output_back, targets_back), hidden_list_back 


  def init_hidden(self, bsz):
    weight = next(self.parameters()).data
    return (Variable(weight.new(self.config['encoder']['n_layers'], bsz, self.config['encoder']['hidden_dim']).zero_()),
            Variable(weight.new(self.config['encoder']['n_layers'], bsz, self.config['encoder']['hidden_dim']).zero_()))

  def save_model(self, path):
    torch.save(self.token_embedder.state_dict(), os.path.join(path, 'token_embedder.pkl'))
    torch.save(self.gaz_embedder.state_dict(), os.path.join(path, 'gaz_embedder.pkl'))     
    torch.save(self.encoder.state_dict(), os.path.join(path, 'encoder.pkl'))
    torch.save(self.classify_layer.state_dict(), os.path.join(path, 'classifier.pkl'))

  def load_model(self, path):
    self.token_embedder.load_state_dict(torch.load(os.path.join(path, 'token_embedder.pkl')))
    self.gaz_embedder.load_state_dict(torch.load(os.path.join(path, 'gaz_embedder.pkl')))
    self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pkl')))
    self.classify_layer.load_state_dict(torch.load(os.path.join(path, 'classifier.pkl')))

if __name__ == "__main__":
  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,format=log_format, datefmt='%m/%d %I:%M:%S %p', filemode='w')
  fh = logging.FileHandler('./log.txt')
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

  if len(sys.argv) > 1 and sys.argv[1] == 'train':
    train()
  elif len(sys.argv) > 1 and sys.argv[1] == 'test':
    test()
  else:
    print('Usage: {0} [train|test] [options]'.format(sys.argv[0]), file=sys.stderr)
