# v0 -> Adds article paragrpahs to 01_facts v4
# v1 -> Moves generic functions to forward section for debugging

# Imports
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoModel

#%% DataClass definition
class ECHR2_dataset(Dataset):
    def __init__(self, data_df):
        self.facts_ids = data_df['facts_ids']
        self.facts_token_types = data_df['facts_token_type']
        self.facts_attn_masks = data_df['facts_att_mask']
        self.echr_ids = data_df['echr_ids']
        self.echr_token_types = data_df['echr_token_type']
        self.echr_attn_masks = data_df['echr_att_mask']
        self.labels = torch.FloatTensor(np.asarray(data_df.label.to_list()))

    def __len__(self):
        return len(self.facts_ids)
        
    def __getitem__(self, idx):
        X_facts_ids = self.facts_ids[idx]
        X_facts_token_types = self.facts_token_types[idx]
        X_facts_attn_maks = self.facts_attn_masks[idx]
        X_echr_ids = self.echr_ids[idx]
        X_echr_token_types = self.echr_token_types[idx]
        X_echr_attn_masks = self.echr_attn_masks[idx]
        Y_labels = self.labels[idx]
        
        return X_facts_ids, X_facts_token_types, X_facts_attn_maks,\
            X_echr_ids, X_echr_token_types, X_echr_attn_masks,Y_labels

#%% Model definition
class ECHR2_model(nn.Module):
            
    def __init__(self, args):
        super(ECHR2_model, self).__init__()

        self.max_n_pars_facts = args.max_n_pars_facts
        self.max_n_pars_echr = args.max_n_pars_echr
        self.h_dim = args.hidden_dim
        self.n_heads = args.n_heads
        self.n_labels = args.num_labels
        self.seq_len = args.seq_len
        self.dropout = args.dropout
                     
        # Bert layer
        self.model_name = 'nlpaueb/legal-bert-small-uncased'
        self.bert_model = AutoModel.from_pretrained(self.model_name)
        
        # Transformer layers
        self.transf_enc_facts = nn.TransformerEncoderLayer(d_model = self.h_dim,
                                                           nhead = self.n_heads)
        self.transf_enc_echr = nn.TransformerEncoderLayer(d_model = self.h_dim,
                                                          nhead = self.n_heads)
    
        # Fully connected output
        self.fc_out = nn.Linear(in_features = (self.max_n_pars_facts + \
                                self.max_n_pars_echr)*self.h_dim,
                                out_features = self.n_labels)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

        # Dropout
        self.drops = nn.Dropout(self.dropout)
           
        # Batch normalization
        self.bn1 = nn.BatchNorm1d((self.max_n_pars_facts + \
                                   self.max_n_pars_echr)*self.h_dim)

    def forward(self, X_facts_ids, X_facts_token_types, X_facts_attn_masks,
                X_echr_ids, X_echr_token_types, X_echr_attn_masks):
        
        # Initialization
        batch_size = X_facts_ids.size()[0]
        device = X_facts_ids.get_device()
        if device == -1: device = 'cpu'
        empty_par_ids = torch.cat([torch.tensor([101,102]),
                                   torch.zeros(self.seq_len-2)]).long() # seq_len
        empty_par_ids = empty_par_ids.repeat(batch_size, 1).to(device)  # batch_size x seq_len

        # BERT paragraph encoding        
        # Encode paragraphs - BERT & generate transfomers masks
        bert_out = {}
        transf_mask_facts = torch.zeros((batch_size,
                                         self.max_num_pars_facts),
                                         dtype=torch.bool).to(device)    # batch_size x max_n_pars
        
        #for idx in range(0, max_num_pars):
        for idx in tqdm(range(0, self.max_num_pars_facts),
                        total = self.max_num_pars_facts,
                        desc = 'Iterating through paragraphs'):
            span_b = self.seq_len * idx
            span_e = self.seq_len * (idx + 1)
            
            # Slice sequence
            ids = X_facts_ids[:, span_b:span_e]                         # batch_size x seq_len
             #token_types = X_facts_token_types[:, span_b:span_e]        # batch_size x seq_len
            attn_masks = X_facts_attn_masks[:, span_b:span_e]           # batch_size x seq_len
            
            # Generate masks for transformer
            equiv = torch.eq(ids, empty_par_ids)                        # batch_size x seq_len
            equiv = equiv.all(dim = 1)                                  # batch_size
            transf_mask_facts[:, idx] = equiv                           # batch_size
            
            # Generate input dict to bert model
            bert_input = {'input_ids': ids.long(),
                           #'token_type_ids': token_types.long(),
                          'attention_mask': attn_masks.long()}
                  
            # Compute bert output
            output = self.bert_model(**bert_input,
                                     output_hidden_states = True)       # Tuple
            bert_out[idx] = output['pooler_output'].unsqueeze(1)        # batch_size x 1 x h_dim
        
        x_facts = torch.cat(list(bert_out.values()), dim=1)             # batch_size x max_n_pars x h_dim
        
        
        # Encode paragraphs - BERT & generate transfomers masks
        bert_out = {}
        transf_mask_echr = torch.zeros((batch_size,
                                        self.max_num_pars_echr),
                                        dtype=torch.bool).to(device)    # batch_size x max_n_pars
        
        #for idx in range(0, max_num_pars):
        for idx in tqdm(range(0, self.max_num_pars_echr),
                        total = self.max_num_pars_echr,
                        desc = 'Iterating through paragraphs'):
            span_b = self.seq_len * idx
            span_e = self.seq_len * (idx + 1)
            
            # Slice sequence
            ids = X_echr_ids[:, span_b:span_e]                          # batch_size x seq_len
             #token_types = X_echr_token_types[:, span_b:span_e]         # batch_size x seq_len
            attn_masks = X_echr_attn_masks[:, span_b:span_e]            # batch_size x seq_len
            
            # Generate masks for transformer
            equiv = torch.eq(ids, empty_par_ids)                        # batch_size x seq_len
            equiv = equiv.all(dim = 1)                                  # batch_size
            transf_mask_echr[:, idx] = equiv                            # batch_size
            
            # Generate input dict to bert model
            bert_input = {'input_ids': ids.long(),
                           #'token_type_ids': token_types.long(),
                          'attention_mask': attn_masks.long()}
                  
            # Compute bert output
            output = self.bert_model(**bert_input,
                                     output_hidden_states = True)       # Tuple
            bert_out[idx] = output['pooler_output'].unsqueeze(1)        # batch_size x 1 x h_dim
        
        x_echr = torch.cat(list(bert_out.values()), dim=1)              # batch_size x max_n_pars x h_dim
        
        # Encode document facts - Transformer
        x_facts = x_facts.transpose(0,1)                                # max_n_pars x batch_size x h_dim
        x_facts = self.transf_enc_facts(x_facts,
                                        src_key_padding_mask =\
                                            transf_mask_facts)          # max_n_pars x batch_size x h_dim
        x_facts = self.drops(x_facts)                                   # max_n_pars x batch_size x h_dim
        x_facts = x_facts.transpose(0,1)                                # batch_size x max_n_pars x h_dim
        
        # Encode document echr - Transformer
        x_echr = x_echr.transpose(0,1)                                  # max_n_pars x batch_size x h_dim
        x_echr = self.transf_enc_echr(x_echr,
                                      src_key_padding_mask =\
                                          transf_mask_echr)             # max_n_pars x batch_size x h_dim
        x_echr = self.drops(x_echr)                                     # max_n_pars x batch_size x h_dim
        x_echr = x_echr.transpose(0,1)                                  # batch_size x max_n_pars x h_dim
        
        # Concatenate fact and echr encodings
        x = torch.cat([x_facts, x_echr], dim = 1)                       # batch_size x (max_n_pars_facts + max_n_pars_echr) x h_dim
        
        # Multi-label classifier      
        x = x.reshape(-1,(self.max_n_pars_facts + \
                          self.max_n_pars_echr)*self.h_dim)             # batch_size x ((max_n_pars_facts + max_n_pars_echr) x h_dim)
        x = self.bn1(x)                                                 # batch_size x ((max_n_pars_facts + max_n_pars_echr) x h_dim)
        x = self.fc_out(x)                                              # batch_size x n_lab
        x = self.sigmoid(x)                                             # batch_size x n_lab

        return x
