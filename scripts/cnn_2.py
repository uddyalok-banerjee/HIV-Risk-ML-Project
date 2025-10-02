#!/usr/bin/env python3

import sys
sys.path.append('../Lib/')

import torch
import torch.nn as nn
import math
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

import pandas as pd
import os, configparser, random, pickle
import data, utils
import statistics
from sklearn.metrics import precision_recall_curve, auc
#from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score, classification_report, multilabel_confusion_matrix, confusion_matrix


#from pytorch_lightning.metrics.classification import PrecisionRecallCurve
#from pytorch_lightning.metrics.functional.classification import auc
# deterministic determinism
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
torch.manual_seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(2020)

# model and model config locations
model_path = 'Model/model.pt'
config_path = 'Model/config.p'

class BagOfEmbeddings(nn.Module):

  def __init__(
    self,
    input_vocab_size,
    output_vocab_size,
    embed_dim,
    hidden_units,
    dropout_rate,
    out_size,
    stride,
    kernel,
    save_config=True):
    """Constructor"""

    super(BagOfEmbeddings, self).__init__()

    self.embed = nn.Embedding(
      num_embeddings=input_vocab_size,
      embedding_dim=embed_dim)

    #self.hidden = nn.Linear(
      #in_features=embed_dim,
      #out_features=hidden_units)

    self.activation = nn.ReLU()

    self.dropout = nn.Dropout(dropout_rate)
    
    #self.kernel_1 = [3]

    self.embed_dim = embed_dim
    self.kernel_1 = [kernel]
    self.out_size = out_size

    self.convs = nn.ModuleList([nn.Conv2d(1, self.out_size, (K, embed_dim)) for K in self.kernel_1])
    
    #self.pool_1 = nn.MaxPool1d(self.kernel_1, int(self.out_size))

    self.hidden_layer = nn.Linear(len(self.kernel_1) * self.out_size, hidden_units)
    self.fc1 = nn.Linear(hidden_units, output_vocab_size)

    # save configuration for loading later

    #self.hash_bucket = set()

    if save_config:
      config = {
        'input_vocab_size': input_vocab_size,
        'output_vocab_size': output_vocab_size,
        'embed_dim': embed_dim,
        'hidden_units': hidden_units,
        'dropout_rate': dropout_rate,
        'out_size': out_size,
        'stride':stride,
        'kernel':kernel}
      pickle_file = open(config_path, 'wb')
      pickle.dump(config, pickle_file)


  def in_features_fc(self):
    out_conv_1=((self.embed_dim - 1 * (self.kernel_1 - 1) -1) / self.stride) + 1
    out_conv_1 = math.floor(out_conv_1)
    out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) -1) / self.stride) + 1
    out_pool_1 = math.floor(out_pool_1)
    return out_pool_1 * self.out_size

  def forward(self, texts, return_hidden=False):
    """Optionally return hidden layer activations"""

    x = self.embed(texts)
    x = x.unsqueeze(1)
    x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
    x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
    x = torch.cat(x, 1)
    x = self.dropout(x)
    x = self.hidden_layer(x)
    out = self.fc1(x)

    if return_hidden:
      return features
    else:
      return out

def make_data_loader(input_seqs, model_outputs, batch_size, partition):
  """DataLoader objects for train or dev/test sets"""

  model_inputs = utils.pad_sequences(input_seqs, max_len=35000) #unique = 37317

  # e.g. transformers take input ids and attn masks
  if type(model_inputs) is tuple:
    tensor_dataset = TensorDataset(*model_inputs, model_outputs)
  else:
    tensor_dataset = TensorDataset(model_inputs, model_outputs)

  # use sequential sampler for dev and test
  if partition == 'train':
    sampler = RandomSampler(tensor_dataset)
  else:
    sampler = SequentialSampler(tensor_dataset)

  data_loader = DataLoader(
    tensor_dataset,
    sampler=sampler,
    batch_size=batch_size)

  return data_loader

def fit(model, train_loader, val_loader, n_epochs, learning_rate, wei):
  """Training routine"""
  print(torch.cuda.is_available())
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = nn.DataParallel(model)
  model.to(device)
  print(model)

  def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(count_parameters(model))

  criterion = nn.BCEWithLogitsLoss()

  optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate)

  best_loss = float('inf')
  best_pr_auc = 0
  optimal_epochs = 0

  for epoch in range(1, n_epochs + 1):

    model.train()
    train_loss, num_train_steps = 0, 0

    for batch in train_loader:

      optimizer.zero_grad()

      batch = tuple(t.to(device) for t in batch)
      batch_inputs, batch_outputs = batch
      logits = model(batch_inputs)
      """
      SUD = torch.tensor([31,34,36]).to(device)
      logist_SUD = torch.index_select(logits, 1, SUD)
      batch_output_SUD = torch.index_select(batch_outputs, 1, SUD)
      loss_SUD = criterion(logist_SUD, batch_output_SUD)

      
      nonSUD = torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32,33,35,37,38,39,40,41,42,43,44,45]).to(device)
      
      logist_nSUD = torch.index_select(logits, 1, nonSUD)
      batch_output_nSUD = torch.index_select(batch_outputs, 1, nonSUD)
      loss_nSUD = criterion(logist_nSUD, batch_output_nSUD)

      loss = (wei * loss_SUD) + ((1-wei) * loss_nSUD)
      
      SUD_A = torch.tensor([0]).to(device)
      SUD_O = torch.tensor([1]).to(device)
      SUD_NO = torch.tensor([2]).to(device)
      logits_A = torch.index_select(logits, 1, SUD_A)
      batch_output_A = torch.index_select(batch_outputs, 1, SUD_A)

      logits_O = torch.index_select(logits, 1, SUD_O)
      batch_output_O = torch.index_select(batch_outputs, 1, SUD_O)
      logits_NO = torch.index_select(logits, 1, SUD_NO)
      batch_output_NO = torch.index_select(batch_outputs, 1, SUD_NO)
      
      loss_A = criterion(logits_A, batch_output_A)
      loss_O = criterion(logits_O, batch_output_O)
      loss_NO = criterion(logits_NO, batch_output_NO)
      loss = ((2/3) * loss_A) + loss_O + (2 * loss_NO)
      """
      loss = criterion(logits, batch_outputs)
      loss.backward()
      
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()

      train_loss += loss.item()
      num_train_steps += 1

    av_tr_loss = train_loss / num_train_steps
    val_loss, pr_auc, y_true, y_pred, pr_auc_L, roc_L = evaluate(model, val_loader)
    print('ep: %d, steps: %d, tr loss: %.4f, val loss: %.4f, pr auc: %.4f' % \
          (epoch, num_train_steps, av_tr_loss, val_loss, pr_auc))

    if pr_auc > best_pr_auc:
      print('pr auc improved, saving model...')
      counts = 0
      torch.save(model.state_dict(), model_path)
      best_pr_auc = pr_auc
      optimal_epochs = epoch
      y_true_a = y_true
      y_pred_a = y_pred
      pr_auc_L_a = pr_auc_L
      roc_L_a = roc_L

    else:
      counts = counts + 1
      if counts == 8:
        print('pr auc did not imporve')  
        return best_pr_auc, optimal_epochs, y_true_a, y_pred_a, pr_auc_L_a, roc_L_a

  return best_pr_auc, optimal_epochs, y_true_a, y_pred_a, pr_auc_L, roc_L_a

def evaluate(model, data_loader):
  """Evaluation routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  #model = nn.DataParallel(model)
  model.to(device)
  
  criterion = nn.BCEWithLogitsLoss()
  total_loss, num_steps = 0, 0

  model.eval()
  y_true = torch.tensor([], dtype=torch.float, device=device)
  all_output = torch.tensor([], device=device)

  for batch in data_loader:

    batch = tuple(t.to(device) for t in batch)
    batch_inputs, batch_outputs = batch

    with torch.no_grad():
      logits = model(batch_inputs)
      loss = criterion(logits, batch_outputs)
      #print(logits)
      y_true = torch.cat((y_true, batch_outputs), 0)
      all_output = torch.cat((all_output, logits), 0)

    total_loss += loss.item()
    num_steps += 1

  y_true = y_true.cpu().numpy()
  y_pred = all_output.cpu().numpy()
  y_pred_prob = torch.sigmoid(all_output).cpu().numpy()
  are_L = []
  roc_L = []
  are_a = []
  for i in range(0, 1):
    y_t = y_true[:,i]
    y_prob = y_pred_prob[:,i]

    if i in [0]:
      precision, recall, _ = precision_recall_curve(y_true=y_t, probas_pred=y_prob)
      are = auc(recall, precision)
      roc_a = roc_auc_score(y_t, y_prob)
      roc_L.append(roc_a)
      are_L.append(are)
      are_a.append(are)

  av_PR_AUC = statistics.mean(are_a)
  av_loss = total_loss / num_steps
  y_pred_pro = y_pred_prob[:,[0]]
  y_tru = y_true[:,[0]]
  return av_loss, av_PR_AUC, y_tru, y_pred_pro, are_L, roc_L

 
def main():
  """My main main"""

  dp = data.DatasetProvider(
    os.path.join(base, cfg.get('data', 'cuis')),
    os.path.join(base, cfg.get('data', 'codes')),
    cfg.get('args', 'cui_vocab_size'),
    cfg.get('args', 'code_vocab_size'))

  in_seqs, out_seqs = dp.load_as_sequences()

  tr_in_seqs, val_in_seqs, tr_out_seqs, val_out_seqs = train_test_split(
    in_seqs, out_seqs, test_size=0.10, random_state=2020)

  print('loaded %d training and %d validation samples' % \
        (len(tr_in_seqs), len(val_in_seqs)))

  max_cui_seq_len = max(len(seq) for seq in tr_in_seqs)
  print('longest cui sequence:', max_cui_seq_len)

  max_code_seq_len = max(len(seq) for seq in tr_out_seqs)
  print('longest code sequence:', max_code_seq_len)

  train_loader = make_data_loader(
    tr_in_seqs,
    utils.sequences_to_matrix(tr_out_seqs, len(dp.output_tokenizer.stoi)),
    cfg.getint('model', 'batch'),
    'train')

  val_loader = make_data_loader(
    val_in_seqs,
    utils.sequences_to_matrix(val_out_seqs, len(dp.output_tokenizer.stoi)),
    cfg.getint('model', 'batch'),
    'dev')

  model = BagOfEmbeddings(
    input_vocab_size=len(dp.input_tokenizer.stoi),
    output_vocab_size=len(dp.output_tokenizer.stoi),
    embed_dim=cfg.getint('model', 'embed'),
    hidden_units=cfg.getint('model', 'hidden'),
    dropout_rate=cfg.getfloat('model', 'dropout'))

  best_loss, optimal_epochs = fit(
    model,
    train_loader,
    val_loader,
    cfg.getint('model', 'epochs'))
  print('best loss %.4f after %d epochs' % (best_loss, optimal_epochs))


def mainRan():

  embed_dim = cfg.get('model', 'embed').split(", ")
  hidden_units=cfg.get('model', 'hidden').split(", ")
  dropout_rate=cfg.get('model', 'dropout').split(", ")
  batch_size = cfg.get('model', 'batch').split(", ")
  search_count = cfg.getint('args', 'search_count')
  learning_rate = cfg.get('model', 'lr').split(", ")
  wei = cfg.get('model','weight').split(", ")
  out_size = cfg.get('model','out_size').split(", ")
  stride = cfg.get('model','stride').split(", ")
  kernel = cfg.get('model','kernel').split(", ")
  dp = data.DatasetProvider(
    os.path.join(base, cfg.get('data', 'cuis')),
    os.path.join(base, cfg.get('data', 'codes')),
    cfg.get('args', 'cui_vocab_size'),
    cfg.get('args', 'code_vocab_size'))

  in_seqs, out_seqs, enc = dp.load_as_sequences()

  #tr_in_seqs, val_in_seqs, tr_out_seqs, val_out_seqs = train_test_split(
    #in_seqs, out_seqs, test_size=0.10, random_state=2020)
  #for x in in_seqs:
    #print(len(x))
  tuple_l = list(zip(in_seqs, out_seqs, enc))

  df_split = pd.read_csv('/raid/SUIT/code/HIV_SUD/Files/HIV_final_dataset_split.csv', dtype={'MRN':str})
  df_split = df_split[df_split["split"].isin([1,2])]
  df_split_val = df_split[df_split["split"] == 2]

  val_in_seqs = []
  val_out_seqs = []
  tr_in_seqs = []
  tr_out_seqs = []

  for x in tuple_l:
    enc = x[2]
    if str(enc) in (df_split_val["MRN"].tolist()):
      val_in_seqs.append(x[0])
      val_out_seqs.append(x[1])
    else:
      tr_in_seqs.append(x[0])
      tr_out_seqs.append(x[1])

  print('loaded %d training and %d validation samples' % \
        (len(tr_in_seqs), len(val_in_seqs)))

  #print(tr_in_seqs)
  #for x in tr_in_seqs:
    #print(len(x))
  max_cui_seq_len = max(len(seq) for seq in tr_in_seqs)
  print('longest cui sequence:', max_cui_seq_len)

  max_code_seq_len = max(len(seq) for seq in tr_out_seqs) 
  print('longest code sequence:', max_code_seq_len)

  n = 1
  result_t = []
  hash_bucket = set()
  while (n <= search_count):
    torch.manual_seed(2020)
    torch.backends.cudnn.deterministic = True
    embed_d = int(random.choice(embed_dim))
    hidden_u = int(random.choice(hidden_units))
    dropout_r = float(random.choice(dropout_rate))
    batch_s = int(random.choice(batch_size))
    learning_R = float(random.choice(learning_rate))
    wei_R = float(random.choice(wei))
    out_s = int(random.choice(out_size))
    stri = int(random.choice(stride))
    ker = int(random.choice(kernel))
    hash_num = hash((embed_d, hidden_u, dropout_r, batch_s, learning_R, wei_R, out_s, stri, ker))
    
    if hash_num not in hash_bucket:
      print("Iteration:", n)
      print("Embed_D:", str(embed_d))
      print("Hidden_U:", str(hidden_u))
      print("Dropout_R:", str(dropout_r))
      print("Batch_S:", str(batch_s))
      print("Learning_R:", str(learning_R))
      print("Weight:", str(wei_R))
      print("Out_size:",str(out_s))
      #print("Stride:",str(stri))
      print("Kernel:",str(ker))
      hash_bucket.add(hash_num)
      n = n + 1

      train_loader = make_data_loader(
        tr_in_seqs,
        utils.sequences_to_matrix(tr_out_seqs, len(dp.output_tokenizer.stoi)),
        batch_s,
        'train')

      val_loader = make_data_loader(
        val_in_seqs,
        utils.sequences_to_matrix(val_out_seqs, len(dp.output_tokenizer.stoi)),
        batch_s,
        'dev')

      model = BagOfEmbeddings(
        input_vocab_size=len(dp.input_tokenizer.stoi),
        output_vocab_size=len(dp.output_tokenizer.stoi),
        embed_dim=embed_d,
        hidden_units=hidden_u,
        dropout_rate=dropout_r,
        out_size=out_s,
        stride=stri,
        kernel=ker)

      best_pr_auc, optimal_epochs, y_true, y_prob, pr_auc_L, roc_L = fit(
        model,
        train_loader,
        val_loader,
        cfg.getint('model', 'epochs'),
        learning_R,
        wei_R)
      
      thresh = 0.5
      #y_pred = np.array([[1 if i > thresh else 0 for i in j] for j in y_prob])
      y_prob = y_prob[:,0]
      y_true = y_true[:,0]
      y_pred = np.array([1 if j > thresh else 0 for j in y_prob])
      print(classification_report(y_true, y_pred))
      print(confusion_matrix(y_true, y_pred))
      print('best pr auc %.4f after %d epochs' % (best_pr_auc, optimal_epochs))
      print()
      print("AUC:")
      print(pr_auc_L)
      print("ROC:")
      print(roc_L)
      print('best pr auc %.4f after %d epochs' % (best_pr_auc, optimal_epochs))
      print()

      result_t.append((embed_d, hidden_u, dropout_r, learning_R, wei_R, optimal_epochs, best_pr_auc))
      #if n == 1:
      #    sys.exit()
    else:
      continue
  
  df_result = pd.DataFrame(result_t, columns=["EMBED_D", "HIDDEN_UNITS", "DROPOUT", "LEARNING_RATE","Weights_SUD", "OPTIMAL_EPOCH", "BEST_PR_AUC"])
  df_result.to_csv("Random_Search_Results/CNN_ADAM_COKE_FULL.csv", sep="|", index=False)  


if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  
  mainRan()
