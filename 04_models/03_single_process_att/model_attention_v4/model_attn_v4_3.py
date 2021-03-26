# v3 -> Uses article paragraphs instead of full document
# v4 -> Uses two-way co-attention
# v4.2 -> Includes masking
# v4.3 -> Includes paragraph annotations

#%% Imports

import torch
import torch.nn as nn
from torch.utils.data import Dataset

#%% DataClass definition

class ECHR_dataset(Dataset):
    def __init__(self, data_df):
        self.article_tensor = torch.LongTensor(data_df['article_pars_ids'].to_list())
        self.cases_tensor = torch.LongTensor(data_df['case_texts_ids'].to_list())
        self.violated_pars_tensor = torch.FloatTensor(data_df['violated_pars'].to_list())
        self.outcome_tensor = torch.Tensor(data_df['outcome'].to_list())
        
    def __len__(self):
        return self.outcome_tensor.size()[0]
        
    def __getitem__(self, idx):
        X_article = self.article_tensor[idx, :]
        X_cases = self.cases_tensor[idx, :]
        X_violated_pars = self.violated_pars_tensor[idx, :]
        Y = self.outcome_tensor[idx]
        
        return X_article, X_cases, X_violated_pars, Y

#%% Model definition

class ECHR_model(nn.Module):
            
    def __init__(self, args, pretrained_embeddings):
        super(ECHR_model, self).__init__()

        self.num_layers = 1
        self.dropout = args.dropout
        self.input_size = args.embed_dim
        self.h_dim = args.hidden_dim
        self.output_size = 1
        self.att_dim = args.hidden_dim * 2
        self.num_passages = args.num_passages
        self.num_par_arts = args.num_par_arts
        self.seq_len = args.seq_len
        self.query_att_art_par = nn.Parameter(torch.randn((1, self.att_dim), requires_grad = True))
        self.query_att_case_pass = nn.Parameter(torch.randn((1, self.att_dim), requires_grad = True))

        # Embedding
        self.embed = nn.Embedding.from_pretrained(pretrained_embeddings)
        
        # Dropout
        self.drops = nn.Dropout(self.dropout)
        
        # Encode article
        self.lstm_art_par = nn.LSTM(input_size = self.input_size,
                                    hidden_size = self.h_dim,
                                    num_layers = self.num_layers,
                                    bidirectional = True,
                                    batch_first = True)      
        
        # Encode case passages
        self.lstm_case_pass = nn.LSTM(input_size = self.input_size,
                                      hidden_size = self.h_dim,
                                      num_layers = self.num_layers,
                                      bidirectional = True,
                                      batch_first = True)
        
        # Fully connected query vector
        self.fc_query = nn.Linear(in_features = self.h_dim * 2,
                                  out_features = self.att_dim)
        
        # Fully connected projection article paragraph 1
        self.fc_proj_art_par_1 = nn.Linear(in_features = self.h_dim * 2,
                                           out_features = self.att_dim)

        # Fully connected projection article paragraph 2
        self.fc_proj_art_par_2 = nn.Linear(in_features = self.h_dim * 2,
                                           out_features = self.att_dim)

        # Fully connected projection case passage
        self.fc_proj_case_pass = nn.Linear(in_features = self.h_dim * 2,
                                           out_features = self.att_dim)

        # Fully connected output
        self.fc_out = nn.Linear(in_features = self.h_dim * 2,
                                out_features = self.output_size)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, X_art, X_case):
        INF = 1e12
        device = X_art.get_device()
        batch_size = X_art.size()[0]
        bilstm_b = self.h_dim
        bilstm_e = self.h_dim * 2

        # Generate article masks
        mask_art = torch.ones([batch_size, self.num_par_arts]).to(device)  # batch_size x n_pars  
        for sample_idx in range(batch_size):
            for pos_idx in range(self.num_par_arts):
                idx_beg = pos_idx*self.seq_len
                idx_end = (pos_idx + 1)*self.seq_len
                art_par = X_art[sample_idx, idx_beg:idx_end]
                if all([x == 0 for x in art_par.tolist()]):
                    mask_art[sample_idx,pos_idx] = 0

        # Generate case passage masks
        mask_case = torch.ones([batch_size, self.num_passages]).to(device) # batch_size x n_passages
        for sample_idx in range(batch_size):
            for pos_idx in range(self.num_passages):
                idx_beg = pos_idx*self.seq_len
                idx_end = (pos_idx + 1)*self.seq_len
                passage = X_case[sample_idx, idx_beg:idx_end]
                if all([x == 0 for x in passage.tolist()]):
                    mask_case[sample_idx,pos_idx] = 0
   
        # Embedding
        x_art = self.embed(X_art)                                          # batch_size x seq_len x embed_dim
        x_case = self.embed(X_case)                                        # batch_size x (seq_len x n_passages) x embed_dim
        
        # Article paragraph encoding
        x_art_par_dict = {}
        
        for idx in range(0, self.num_par_arts):
            span_b = self.seq_len * idx
            span_e = self.seq_len * (idx + 1)
            self.lstm_art_par.flatten_parameters()
            x_aux = self.lstm_art_par(x_art[:,span_b:span_e,:])            # Tuple (len = 2)
            x_aux_fwd = x_aux[0][:, -1, 0:bilstm_b]                        # batch_size x hidden_dim
            x_aux_bkwd = x_aux[0][:, 0, bilstm_b:bilstm_e]                 # batch_size x hidden_dim
            x_aux = torch.cat((x_aux_fwd, x_aux_bkwd), dim = 1)            # batch_size x (hidden_dim x 2)
            x_aux = self.drops(x_aux)                                      # batch_size x (hidden_dim x 2)
            x_art_par_dict[idx] = x_aux.unsqueeze(1)                       # batch_size x 1 x (hidden_dim x 2)
        
        x_art_par = torch.cat(list(x_art_par_dict.values()),dim=1)         # batch_size x n_pars x (hidden_dim x 2)
        
        # Article paragraph encoding - attention step 1
        neg_inf = torch.ones(mask_art.size()).fill_(-INF).to(device)       # batch_size x n_pars
        projection = torch.tanh(self.fc_proj_art_par_1(x_art_par))         # batch_size x n_pars x att_dim
        query_att_art_par = torch.transpose(self.query_att_art_par, 0, 1)  # att_dim x 1
        alpha_1 = torch.matmul(projection, query_att_art_par)              # batch_size x n_pars x 1
        alpha_1 = alpha_1.squeeze(2)                                       # batch_size x n_pars
        masked_alpha_1 = torch.where(mask_art>0, alpha_1, neg_inf)         # batch_size x n_pars
        masked_alpha_1 = torch.softmax(masked_alpha_1, dim = 1)            # batch_size x n_pars
        masked_alpha_1 = masked_alpha_1.unsqueeze(2)                       # batch_size x n_pars x 1
        att_output = x_art_par * masked_alpha_1                            # batch_size x n_pars x (hidden_dim x 2)
        query_1 = torch.sum(att_output, axis = 1).unsqueeze(1)             # batch_size x 1 x (hidden_dim x 2)            
              
        # Case sentence encoding
        x_case_pass_dict = {}
        
        for idx in range(0, self.num_passages):
            span_b = self.seq_len * idx
            span_e = self.seq_len * (idx + 1)
            self.lstm_case_pass.flatten_parameters()
            x_aux = self.lstm_case_pass(x_case[:,span_b:span_e,:])         # Tuple (len = 2)
            x_aux_fwd = x_aux[0][:, -1, 0:bilstm_b]                        # batch_size x hidden_dim
            x_aux_bkwd = x_aux[0][:, 0, bilstm_b:bilstm_e]                 # batch_size x hidden_dim
            x_aux = torch.cat((x_aux_fwd, x_aux_bkwd), dim = 1)            # batch_size x (hidden_dim x 2)
            x_aux = self.drops(x_aux)                                      # batch_size x (hidden_dim x 2)
            x_case_pass_dict[idx] = x_aux.unsqueeze(1)                     # batch_size x 1 x (hidden_dim x 2)
        
        x_case_pass = torch.cat(list(x_case_pass_dict.values()),dim=1)     # batch_size x n_passages x (hidden_dim x 2)
               
        # Case document encoding - attention 
        neg_inf = torch.ones(mask_case.size()).fill_(-INF).to(device)      # batch_size x n_pars
        projection = torch.tanh(self.fc_proj_case_pass(x_case_pass))       # batch_size x n_passages x att_dim
        query_1 = torch.transpose(query_1, 1, 2)                           # batch_size x (hidden_dim x 2) x 1
        alpha_2 = torch.bmm(projection, query_1)                           # batch_size x n_passages x 1
        alpha_2 = alpha_2.squeeze(2)                                       # batch_size x n_passages
        masked_alpha_2 = torch.where(mask_case>0, alpha_2, neg_inf)        # batch_size x n_passages
        masked_alpha_2 = torch.softmax(masked_alpha_2, dim = 1)            # batch_size x n_passages
        masked_alpha_2 = masked_alpha_2.unsqueeze(2)                       # batch_size x n_passages x 1
        att_output = x_case_pass * masked_alpha_2                          # batch_size x n_passages x (hidden_dim x 2)
        query_2 = torch.sum(att_output, axis = 1).unsqueeze(1)             # batch_size x 1 x (hidden_dim x 2)
        
        # Article paragraph encoding - attention step 2
        neg_inf = torch.ones(mask_art.size()).fill_(-INF).to(device)       # batch_size x n_pars
        projection = torch.tanh(self.fc_proj_art_par_2(x_art_par))         # batch_size x n_pars x att_dim
        query_2 = torch.transpose(query_2, 1, 2)                           # batch_size x (hidden_dim x 2) x 1
        alpha_3 = torch.bmm(projection, query_2)                           # batch_size x n_pars x 1
        alpha_3 = alpha_3.squeeze(2)                                       # batch_size x n_pars
        masked_alpha_3 = torch.where(mask_art>0, alpha_3, neg_inf)         # batch_size x n_pars
        masked_alpha_3 = torch.softmax(masked_alpha_3, dim = 1)            # batch_size x n_pars
        masked_alpha_3 = masked_alpha_3.unsqueeze(2)                       # batch_size x n_pars x 1
        att_output = x_art_par * masked_alpha_3                            # batch_size x n_pars x (hidden_dim x 2)
        att_output = torch.sum(att_output, axis = 1).unsqueeze(1)          # batch_size x 1 x (hidden_dim x 2)            
        
        # Fully conected output
        x = self.fc_out(att_output)                                        # batch size x output_size
        
        # Sigmoid function
        x = self.sigmoid(x)                                                # batch size x output_size
        
        return x, masked_alpha_2, masked_alpha_3
