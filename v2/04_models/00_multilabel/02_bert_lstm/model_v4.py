# v2 -> Tries to use stacked tensors
# v3 -> Uses flat tensors
# v4 -> Decopuled from train

# Imports
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel

#%% DataClass definition
class ECHR2_dataset(Dataset):
    def __init__(self, data_df):
        self.facts_ids = data_df['facts_ids']
        self.facts_token_types = data_df['facts_token_type']
        self.facts_attn_masks = data_df['facts_att_mask']
        self.labels = torch.FloatTensor(np.asarray(data_df.labels.to_list()))
        
    def __len__(self):
        return len(self.facts_ids)
        
    def __getitem__(self, idx):
        X_facts_ids = self.facts_ids[idx]
        X_facts_token_types = self.facts_token_types[idx]
        X_facts_attn_maks = self.facts_attn_masks[idx]
        Y_labels = self.labels[idx]
        
        return X_facts_ids, X_facts_token_types, X_facts_attn_maks, Y_labels

#%% Model definition
class ECHR2_model(nn.Module):
            
    def __init__(self, args):
        super(ECHR2_model, self).__init__()

        self.max_n_pars = args.max_n_pars
        self.h_dim = args.hidden_dim
        self.h_dim_lstm = 200
        self.n_heads = args.n_heads
        self.n_labels = args.num_labels
        self.seq_len = args.seq_len
        self.dropout = args.dropout
        self.num_layers = 1
                     
        # Bert model
        self.model_name = 'nlpaueb/legal-bert-small-uncased'
        self.bert_model = AutoModel.from_pretrained(self.model_name)
        # Freeze bert parameters
        for parameter in self.bert_model.parameters():
            parameter.requires_grad = False
        
        # Transformer layer
        self.transf_enc = nn.TransformerEncoderLayer(d_model = self.h_dim,
                                                     nhead = self.n_heads)
    
        # LSTM layer
        self.lstm = nn.LSTM(input_size = self.h_dim,
                            hidden_size = self.h_dim_lstm,
                            num_layers = self.num_layers,
                            bidirectional = True,
                            batch_first = True)

        # Fully connected output
        self.fc_out = nn.Linear(in_features = self.h_dim_lstm*2,
                                out_features = self.n_labels)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

        # Dropout
        self.drops = nn.Dropout(self.dropout)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(self.h_dim_lstm*2)
            
    def forward(self, X_facts_ids, X_facts_token_types, X_facts_attn_masks):
        # Encode paragraphs - BERT
        bert_out = {}
        
        for idx in tqdm(range(0, self.max_n_pars),
                        desc = 'Iterating through paragraphs'):

#        for idx in range(0, self.max_n_pars):
            span_b = self.seq_len * idx
            span_e = self.seq_len * (idx + 1)
            
            # Slice sequence
            facts_ids = X_facts_ids[:, span_b:span_e]                   # batch_size x seq_len
            #facts_token_types = X_facts_token_types[:, span_b:span_e]  # batch_size x seq_len
            facts_attn_masks = X_facts_attn_masks[:, span_b:span_e]     # batch_size x seq_len
            
            # Generate input dict to bert model
            bert_input = {'input_ids': facts_ids.long(),
            #              'token_type_ids': facts_token_types.long(),
                          'attention_mask': facts_attn_masks.long()}
                  
            # Compute bert output
            output = self.bert_model(**bert_input, output_hidden_states = True)
            bert_out[idx] = output['pooler_output'].unsqueeze(1)        # batch_size x 1 x h_dim
        
        x = torch.cat(list(bert_out.values()), dim=1)                   # batch_size x max_n_pars x h_dim
        
        # Encode document - LSTM
        self.lstm.flatten_parameters()
        x = self.lstm(x)                                                # Tuple (len = 2)
        x_fwd = x[0][:, -1, 0:self.h_dim_lstm]                          # batch_size x hidden_dim
        x_bkwd = x[0][:, 0, self.h_dim_lstm:self.h_dim_lstm*2]          # batch_size x hidden_dim
        x = torch.cat((x_fwd, x_bkwd), dim = 1)                         # batch_size x (hidden_dim x 2)
        x = self.drops(x)                                               # batch_size x (hidden_dim x 2)
        
        # Multi-label classifier
        x = self.bn1(x)                                                 # batch_size x (hidden_dim x 2)
        x = self.fc_out(x)                                              # batch_size x n_lab
        x = self.sigmoid(x)                                             # batch_size x n_lab

        return x
