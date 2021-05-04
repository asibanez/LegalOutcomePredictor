# v2 -> Tries to use stacked tensors
# v3 -> Uses flat tensors
# v4 -> Decopuled from train

# Imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


#%% DataClass definition
class ECHR2_dataset(Dataset):
    def __init__(self, data_df):
        self.bert_encoding = data_df['bert_encoding']
        self.transf_mask = data_df['transf_mask']
        self.labels = torch.FloatTensor(np.asarray(data_df.labels.to_list()))
        
    def __len__(self):
        return len(self.bert_encoding)
        
    def __getitem__(self, idx):
        X_bert_encoding = self.bert_encoding[idx]
        X_transf_mask = self.transf_mask[idx]
        Y_labels = self.labels[idx]
        
        return X_bert_encoding, X_transf_mask, Y_labels

#%% Model definition
class ECHR2_model(nn.Module):
            
    def __init__(self, args):
        super(ECHR2_model, self).__init__()

        self.max_n_pars = args.max_n_pars
        self.h_dim = args.hidden_dim
        self.n_heads = args.n_heads
        self.n_labels = args.num_labels
        self.seq_len = args.seq_len
        self.dropout = args.dropout
        
        # Transformer layer
        self.transf_enc = nn.TransformerEncoderLayer(d_model = self.h_dim,
                                                     nhead = self.n_heads)
    
        # Fully connected 1
        self.fc_1 = nn.Linear(in_features = self.h_dim,
                              out_features = int(self.h_dim/2))
        
        # Fully connected output
#        self.fc_out = nn.Linear(in_features = int(self.h_dim/2),
#                                out_features = self.n_labels)

        self.fc_out = nn.Linear(in_features = int(self.h_dim),
                                out_features = self.n_labels)

        # Max pooling
        self.max_pool = nn.MaxPool1d(kernel_size = self.max_n_pars)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

        # Dropout
        self.drops = nn.Dropout(self.dropout)
           
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(self.h_dim)
#        self.bn2 = nn.BatchNorm1d(int(self.h_dim/2))
        self.bn2 = nn.BatchNorm1d(int(self.h_dim))
        
 
    def forward(self, X_bert_encoding, X_transf_mask):
        # X_bert_encoding                                               # batch_size x max_n_pars x h_dim
        # X_transf_mask                                                 # batch_size x max_n_pars
        
        # Encode document - Transformer
        x = X_bert_encoding.transpose(0,1)                              # max_n_pars x batch_size x h_dim
        x = self.transf_enc(x, src_key_padding_mask = X_transf_mask)    # max_n_pars x batch_size x h_dim
        x = self.drops(x)                                               # max_n_pars x batch_size x h_dim
        x = x.transpose(0,1)                                            # batch_size x max_n_pars x h_dim
        
        # Fully connected 1
#        x = x.transpose(1,2)                                            # batch_size x h_dim x max_n_pars
#        x = self.bn1(x)                                                 # batch_size x max_n_pars x h_dim
#        x = x.transpose(1,2)                                            # batch_size x max_n_pars x h_dim
#        x = self.fc_1(x)                                                # batch_size x max_n_pars x h_dim/2
#        x = self.drops(x)                                               # batch_size x max_n_pars x h_dim/2

        # Max pooling
        x = x.transpose(1,2)                                            # batch_size x h_dim x max_n_pars
        x = self.max_pool(x)                                            # batch_size x h_dim x 1
        
        # Multi-label classifier
        x = x.squeeze(2)                                                # batch_size x h_dim
        x = self.bn2(x)                                                 # batch_size x h_dim
        x = self.fc_out(x)                                              # batch_size x n_lab
        x = self.sigmoid(x)                                             # batch_size x n_lab

        return x
