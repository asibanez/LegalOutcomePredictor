# Imports
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
#import inspect
#lines = inspect.getsource(AutoModelForSequenceClassification)

#%% DataClass definition
class ECHR2_dataset(Dataset):
    def __init__(self, data_df):
        #self.facts = np.asarray(data_df['facts'].to_list())
        #self.facts = data_df.facts
        #self.labels = np.asarray(data_df.labels.to_list())
        self.dataset = data_df
        
    def __len__(self):
        #return len(self.facts)
        return len(self.dataset)
        
    def __getitem__(self, idx):
        #X_facts = self.facts[idx]
        #Y_labels = self.labels[idx]
        
        samples = self.dataset.iloc[idx]
        
        #return X_facts, Y_labels
        return samples

#%% Model definition
class ECHR2_model(nn.Module):
            
    def __init__(self, args):
        super(ECHR2_model, self).__init__()

        self.num_pars = 37
        self.max_n_pars = 200
        self.h_dim = 512
        self.n_heads = 8
        self.n_labels = 33
        self.seq_len = 512
        self.dropout = 0.4 #args.dropout
                     
        # Bert model
        self.model_name = 'nlpaueb/legal-bert-small-uncased'
        self.bert_model = AutoModel.from_pretrained(self.model_name)
        
        # Transformer layer
        self.transf_enc = nn.TransformerEncoderLayer(d_model = self.h_dim,
                                                     nhead = self.n_heads)
    
        # Fully connected output
        self.fc_out = nn.Linear(in_features = self.h_dim,
                                out_features = self.n_labels)

        # Pooling
        self.max_pool = nn.MaxPool1d(kernel_size = self.max_n_pars)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

        # Dropout
        self.drops = nn.Dropout(self.dropout)
            
    def forward(self, facts):
        # Encode paragraphs - BERT
        bert_out = {}
        num_pars = len(facts)
        for idx, fact in enumerate(facts):
            output = self.bert_model(**fact, output_hidden_states = True)
            #bert_out[idx] = output['last_hidden_state']
            bert_out[idx] = output['pooler_output']             # 1 x h_dim
        x = torch.cat(list(bert_out.values()),dim=0)            # n_pars x h_dim
        
        # Pad paragraphs to max_num_pars
        pad_len = self.max_n_pars - num_pars
        pad = torch.zeros((pad_len, self.h_dim))                # (max_n_pars - n_pars) x h_dim
        x = torch.cat([x, pad], dim = 0)                        # max_n_pars x h_dim
    
        # Compute transformer masks
        transf_mask_false = [False] * num_pars
        transf_mask_true = [True] * pad_len
        transf_mask = torch.BoolTensor(transf_mask_false +
                                       transf_mask_true)        # max_n_pars
        transf_mask = transf_mask.unsqueeze(0)                  # batch_size x max_n_pars
    
        # Encode document - Transformer
        x = x.unsqueeze(0)                                      # batch_size x max_n_pars x h_dim
        x = x.transpose(0, 1)                                   # max_n_pars x batch_size x h_dim
        x = self.transf_enc(x,src_key_padding_mask=transf_mask) # max_n_pars x batch_size x h_dim
        x = x.transpose(0, 1)                                   # batch_size x max_n_pars x h_dim
        
        # Max pooling over paragraphs
        x = x.transpose(1,2)                                    # batch_size x h_dim x max_n_pars
        x = self.max_pool(x)                                    # batch_size x h_dim x 1
        
        # Multi-label classifier
        x = x.squeeze(2)                                        # batch_size x h_dim
        x = self.fc_out(x)                                      # batch_size x n_lab
        x = self.sigmoid(x)                                     # batch_size x n_lab

        return x

#%% Load data
test_set = pd.read_pickle('C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/v2/01_preprocessed/00_toy/model_test.pkl')

#%% Instantiate dataclass & dataloader
test_dataset = ECHR2_dataset(test_set)
test_dl = DataLoader(test_dataset, batch_size = 2,
                     shuffle = False, drop_last = True)

#%% Instantiate model
ECHR2_model = ECHR2_model(None)

#%% Compute predictions
#for X_facts, Y_labels in test_dl:
for samples in test_dl:
    pred = ECHR2_model(samples)