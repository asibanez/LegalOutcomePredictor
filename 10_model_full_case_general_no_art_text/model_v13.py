# v1 ->  Pickle dataframes read with pandas - "Load data" section
# v2 ->  Splits LSTM into 2 (LSTM and case)
# v3 ->  Uses FastText embeddings
# v4 ->  Conversion to token IDs, prunning and padding moved to preprocessing
# v5 ->  Splits cases in 5 different LSTMs
# v6 ->  Sigmoid function introduced.
#        Loss function changed from nn.BCEWithLogitsLoss() to nn.BCE()
#        Dropout added to LSTMS
# v7 ->  Test function added
#        Metric computations added
#        Saves model and results history
# v8 ->  Adds attention layers
# v9 ->  Adds flexible attention and hidden dims to global initialization
# v10 -> Article text removed from model
# v11 -> Article text added to model
# v12 -> Encodes case passages in for loop
# v13 -> Train moved to independent script

#%% Imports

import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Function definition
#%% DataClass definition

class ECHR_dataset(Dataset):
    def __init__(self, data_df):
        self.article_tensor = torch.LongTensor(data_df['article_text'].to_list())
        self.cases_tensor = torch.LongTensor(data_df['case_texts'].to_list())
        self.outcome_tensor = torch.Tensor(data_df['outcome'].to_list())
        
    def __len__(self):
        return self.outcome_tensor.size()[0]
        
    def __getitem__(self, idx):
        X_article = self.article_tensor[idx, :]
        X_cases = self.cases_tensor[idx, :]
        Y = self.outcome_tensor[idx]
        
        return X_article, X_cases, Y

#%% Model definition

class ECHR_model(nn.Module):
    
    def __init__(self, input_size, hidden_dim, output_size, pretrained_embeddings,
                 att_dim, dropout):
        super(ECHR_model, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.dropout = dropout
        self.num_layers = 1
        self.num_passages = 5 #300
        self.seq_len = 512

        # Embedding
        self.embed = nn.Embedding.from_pretrained(pretrained_embeddings)
        
        # Dropout
        self.drops = nn.Dropout(self.dropout)
        
        # Encode article
        self.lstm_art = nn.LSTM(input_size = self.input_size,
                                hidden_size = self.hidden_dim,
                                num_layers = self.num_layers,
                                bidirectional = True,
                                batch_first = True)      
        
        # Encode case senteces
        self.lstm_case_sent = nn.LSTM(input_size = self.input_size,
                                      hidden_size = self.hidden_dim,
                                      num_layers = self.num_layers,
                                      bidirectional = True,
                                      batch_first = True)
        
        # Encode case document
        self.lstm_case_doc = nn.LSTM(input_size = self.hidden_dim * 2,
                                     hidden_size = self.hidden_dim,
                                     num_layers = self.num_layers,
                                     bidirectional = True,
                                     batch_first = True)
                
        # Fully connected
        #self.fc_1 = nn.Linear(in_features = self.hidden_dim * 2 * 2,
        #                      out_features = self.output_size)
        
        self.fc_1 = nn.Linear(in_features = self.hidden_dim * 1 * 2,
                              out_features = self.output_size)
        
        # Sigmoid
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, X_art, X_case):
        # Embedding
        #x_art = self.embed(X_art)                                      # batch_size x seq_len x embed_dim
        x_case = self.embed(X_case)                                    # batch_size x (seq_len x n_passages) x embed_dim
        
        # Article encoding
        #x_art = self.lstm_art(x_art)                                   # Tuple (len = 2)
        #x_art_fwd = x_art[0][:, -1, 0:64]                              # batch_size x hidden_dim
        #x_art_bkwd = x_art[0][:, 0, 64:128]                            # batch_size x hidden_dim
        #x_art = torch.cat((x_art_fwd, x_art_bkwd), dim = 1)            # batch_size x (hidden_dim x 2)
        #x_art = self.drops(x_art)                                      # batch_size x (hidden_dim x 2) 
        
        # Case sentence encoding
        x_case_sent = torch.FloatTensor().to(x_case.device)
        
        for idx in range(0, self.num_passages):
            span_b = self.seq_len * idx
            span_e = self.seq_len * (idx + 1)
            x_aux = self.lstm_case_sent(x_case[:, span_b:span_e, :])  # Tuple (len = 2)
            x_aux_fwd = x_aux[0][:, -1, 0:64]                         # batch_size x hidden_dim
            x_aux_bkwd = x_aux[0][:, 0, 64:128]                       # batch_size x hidden_dim
            x_aux = torch.cat((x_aux_fwd, x_aux_bkwd), dim = 1)       # batch_size x (hidden_dim x 2)
            x_aux = self.drops(x_aux)                                 # batch_size x (hidden_dim x 2)
            x_aux = x_aux.unsqueeze(1)                                # batch_size x 1 x (hidden_dim x 2)
            x_case_sent = torch.cat((x_case_sent, x_aux), dim=1)      # batch_size x n_passages x (hidden_dim x 2)
               
        # Case document encoding
        x_case_doc = self.lstm_case_doc(x_case_sent)                  # Tuple (len = 2)
        x_case_doc_fwd = x_case_doc[0][:, -1, 0:64]                   # batch_size x hidden_dim
        x_case_doc_bkwd = x_case_doc[0][:, 0, 64:128]                 # batch_size x hidden_dim
        x_case_doc = torch.cat((x_case_doc_fwd, x_case_doc_bkwd),
                               dim = 1)                               # batch_size x (hidden_dim x 2)
        x_case_doc = self.drops(x_case_doc)                           # batch_size x (hidden_dim x 2)
        
        # Concatenate  article and passage encodings
        #x = torch.cat((x_art, x_case_doc), dim = 1)                   # batch size x (hidden_dim x 2 x 2)
        x = x_case_doc                   # batch size x (hidden_dim x 2 x 2)
        
        # Fully connected layer
        x = self.fc_1(x)                                              # batch size x output_size
        
        # Sigmoid function
        x = self.sigmoid(x)                                           # batch size x output_size
        
        return x


