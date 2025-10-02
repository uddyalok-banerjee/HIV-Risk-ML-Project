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
from collections import OrderedDict
import pandas as pd
import os, configparser, random, pickle
import data_t, utils
import statistics
from sklearn.metrics import precision_recall_curve, auc
#from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score, classification_report, multilabel_confusion_matrix, confusion_matrix


#from pytorch_lightning.metrics.classification import PrecisionRecallCurve
#from pytorch_lightning.metrics.functional.classification import auc
# deterministic determinism
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
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
    #self.init_weights()

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

  def init_weights(self):
    """Never trust pytorch default weight initialization"""
    torch.nn.init.xavier_uniform_(self.embed.weight)
    torch.nn.init.xavier_uniform_(self.hidden_layer.weight)
    #torch.nn.init.xavier_uniform_(self.fc1.weight)
    torch.nn.init.xavier_uniform_(self.convs[0].weight)
    #torch.nn.init.zeros_(self.embed.bias)
    torch.nn.init.zeros_(self.convs[0].bias)
    torch.nn.init.zeros_(self.hidden_layer.bias)
    #torch.nn.init.zeros_(self.fc1.bias)
    


def make_data_loader(input_seqs, model_outputs, batch_size, partition):
  """DataLoader objects for train or dev/test sets"""
  tot = len(input_seqs)
  model_inputs = utils.pad_sequences(input_seqs, max_len=12000) #unique = 37317

  model_outputs = [0] * tot
  #print(model_outputs)
  # e.g. transformers take input ids and attn masks
  if type(model_inputs) is tuple:
    tensor_dataset = TensorDataset(*model_inputs, model_outputs)
  else:
    tensor_dataset = TensorDataset(model_inputs)

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
      if counts == 5:
        print('pr auc did not imporve')  
        return best_pr_auc, optimal_epochs, y_true_a, y_pred_a, pr_auc_L_a, roc_L_a

  return best_pr_auc, optimal_epochs, y_true_a, y_pred_a, pr_auc_L, roc_L_a

def evaluate(model, data_loader):
  """Evaluation routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  #model = nn.DataParallel(model)
  model.to(device)
  
  #criterion = nn.BCEWithLogitsLoss()
  criterion = nn.CrossEntropyLoss()
  total_loss, num_steps = 0, 0

  model.eval()
  y_true = torch.tensor([], dtype=torch.float, device=device)
  all_output = torch.tensor([], device=device)

  for batch in data_loader:

    #batch = tuple(t.to(device) for t in batch)
    #print(batch)
    batch_inputs = batch[0].to(device)
    #batch_output_argmax = batch_outputs.argmax(1)

    with torch.no_grad():
      logits = model(batch_inputs)
      #loss = criterion(logits, batch_outputs)
      #print(logits)
      #y_true = torch.cat((y_true, batch_outputs), 0)
      all_output = torch.cat((all_output, logits), 0)

    #total_loss += loss.item()
    num_steps += 1

  #y_true = y_true.cpu().numpy()
  #y_pred = all_output.cpu().numpy()
  #print(all_output)
  y_pred_prob = torch.sigmoid(all_output).cpu().numpy()
  """
  are_L = []
  roc_L = []
  are_a = []
  
  for i in range(0, 2):
    #y_t = y_true[:,i]
    y_prob = y_pred_prob[:,i]

    if i in [0, 1]:
      precision, recall, _ = precision_recall_curve(y_true=y_t, probas_pred=y_prob)
      are = auc(recall, precision)
      roc_a = roc_auc_score(y_t, y_prob)
      roc_L.append(roc_a)
      are_L.append(are)
      are_a.append(are)
  
  av_PR_AUC = statistics.mean(are_a)
  av_loss = total_loss / num_steps
  y_pred_pro = y_pred_prob[:,[0,1]]
  y_tru = y_true[:,[0,1]]
  """
  return y_pred_prob


def mainRan():

  dp = data_t.DatasetProvider(
    os.path.join(base, cfg.get('data', 'cuis')),
    os.path.join(base, cfg.get('data', 'codes')),
    cfg.get('args', 'cui_vocab_size'),
    cfg.get('args', 'code_vocab_size'))

  tr_in_seqs, tr_out_seqs, enc = dp.load_as_sequences()

  print('loaded %d training:' % \
        (len(tr_in_seqs)))
  print("Total ENC:")
  print(len(enc))
  #print(tr_in_seqs)
  max_cui_seq_len = max(len(seq) for seq in tr_in_seqs)
  print('longest cui sequence:', max_cui_seq_len)

  #max_code_seq_len = max(len(seq) for seq in tr_out_seqs) 
  #print('longest code sequence:', max_code_seq_len)
  
  batch_s = 64
  val_loader = make_data_loader(
    tr_in_seqs,
    utils.sequences_to_matrix(tr_out_seqs, len(dp.output_tokenizer.stoi)),
    batch_s,
    'dev')
    #load model here

  pkl = open(cfg.get('data','config_pickle'), 'rb')
  config = pickle.load(pkl)
  model = BagOfEmbeddings(
    input_vocab_size=config['input_vocab_size'],
    output_vocab_size=config['output_vocab_size'],
    embed_dim=config['embed_dim'],
    hidden_units=config['hidden_units'],
    dropout_rate=config['dropout_rate'],
    out_size=config['out_size'],
    stride=config['stride'],
    kernel=config['kernel'],
    save_config=False)

  state_dict = torch.load(cfg.get('data','model_file'), map_location=torch.device('cpu'))

  new_state_dict = OrderedDict()
  for k, v in state_dict.items():
    name = k[7:]
    new_state_dict[name] = v
  model.load_state_dict(new_state_dict)
  #y_prob = evaluate(model, val_loader)
  y_prob = evaluate(model, val_loader)
  #thresh = 0.5
  #y_prob = y_prob[:,0]
  #y_true = y_true[:,0]
  #y_pred = np.array([1 if j > thresh else 0 for j in y_prob])
  #print(classification_report(y_true, y_pred))
  #print(confusion_matrix(y_true, y_pred))
  #y_pred = np.array([[1 if i > thresh else 0 for i in j] for j in y_prob])
  #print(classification_report(y_true, y_pred))
  #print(multilabel_confusion_matrix(y_true, y_pred))
  #print()
  #print("AUC:")
  #print(pr_auc_L)
  #print("ROC:")
  #print(roc_L)
  #print()
  #print(y_true)
  #print(y_prob)
  #print(y_prob)
  y_prob = y_prob[:,0]
  df_final = pd.DataFrame({"HSP_ACCOUNT_ID":enc,"Y_PROB":y_prob})
  print(df_final)
  #df_final = pd.concat([df_final, pd.DataFrame(y_prob)],axis=1)
  #df_final.columns = ["HSP_ACCOUNT_ID","Y_prob"]
  #df_final["NHIV_T"] = df_final["NHIV_T"].astype(int)
  #df_final["HIVAQUI_T"] = df_final["HIVAQUI_T"].astype(int)
  #df_final[""] = df_final["HIVTRANS_T"].astype(int)
  #print(df_final[df_final["A_T"] == 1])
  df_final.to_csv("Results/CNN_HIV_FINAL_result_test_24hr_v1.csv",sep=",",index=False)


if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  
  mainRan()
