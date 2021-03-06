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
# v13 -> Train moved to independent script train.py
# v14 -> Added option to include / exclude article text
#        seq_len and num_passages added as variables in train.py
#        fwd and bkwd dimesions computed based on hidden_dim variable
# v15 -> Adds coattention article -> passage 
# v16 -> Improved concatenation of case sentence encodings

#%% Imports

import torch
import torch.nn as nn
from torch.utils.data import Dataset

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
            
    def __init__(self, args, pretrained_embeddings):
        super(ECHR_model, self).__init__()

        self.num_layers = 1
        self.dropout = args.dropout
        self.input_size = args.embed_dim
        self.h_dim = args.hidden_dim
        self.output_size = 1
        self.att_dim = args.att_dim
        self.num_passages = args.num_passages
        self.seq_len = args.seq_len

        # Embedding
        self.embed = nn.Embedding.from_pretrained(pretrained_embeddings)
        
        # Dropout
        self.drops = nn.Dropout(self.dropout)
        
        # Encode article
        self.lstm_art = nn.LSTM(input_size = self.input_size,
                                hidden_size = self.h_dim,
                                num_layers = self.num_layers,
                                bidirectional = True,
                                batch_first = True)      
        
        # Encode case senteces
        self.lstm_case_sent = nn.LSTM(input_size = self.input_size,
                                      hidden_size = self.h_dim,
                                      num_layers = self.num_layers,
                                      bidirectional = True,
                                      batch_first = True)
        
        # Encode case document
        self.lstm_case_doc = nn.LSTM(input_size = self.h_dim * 2,
                                     hidden_size = self.h_dim,
                                     num_layers = self.num_layers,
                                     bidirectional = True,
                                     batch_first = True)
                
        # Fully connected with article text
        self.fc_1 = nn.Linear(in_features = self.h_dim * 2 * 2,
                              out_features = self.output_size)
        

        # Fully connected query vector
        self.fc_query = nn.Linear(in_features = self.h_dim * 2,
                                  out_features = self.att_dim)

        # Fully connected context
        self.fc_context = nn.Linear(in_features = self.h_dim * 2,
                                    out_features = self.att_dim)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, X_art, X_case):
        bilstm_b = self.h_dim
        bilstm_e = self.h_dim * 2

        # Embedding
        x_art = self.embed(X_art)                                      # batch_size x seq_len x embed_dim
        x_case = self.embed(X_case)                                    # batch_size x (seq_len x n_passages) x embed_dim
        
        # Article encoding
        self.lstm_art.flatten_parameters()
        x_art = self.lstm_art(x_art)                                   # Tuple (len = 2)
        x_art_fwd = x_art[0][:, -1, 0:bilstm_b]                        # batch_size x hidden_dim
        x_art_bkwd = x_art[0][:, 0, bilstm_b:bilstm_e]                 # batch_size x hidden_dim
        x_art = torch.cat((x_art_fwd, x_art_bkwd), dim = 1)            # batch_size x (hidden_dim x 2)
        x_art = self.drops(x_art)                                      # batch_size x (hidden_dim x 2) 
        
        # Query vector
        query_v = self.fc_query(x_art).unsqueeze(2)                    # batch_size x att_dim x 1
        
        # Case sentence encoding
        x_case_dict = {}
        
        for idx in range(0, self.num_passages):
            span_b = self.seq_len * idx
            span_e = self.seq_len * (idx + 1)
            self.lstm_case_sent.flatten_parameters()
            x_aux = self.lstm_case_sent(x_case[:,span_b:span_e,:])[0] # batch_size x seq_len x (hidden_dim x 2)
            x_aux = self.drops(x_aux)                                 # batch_size x seq_len x (hidden_dim x 2)
            # Co-attention
            projection = torch.tanh(self.fc_context(x_aux))           # batch_size x seq_len x att_dim
            alpha = torch.bmm(projection, query_v)                    # batch _size x seq_len x 1
            alpha = torch.softmax(alpha, dim = 1)                     # batch_size x seq_len x 1
            att_output = x_aux * alpha                                # batch_size x seq_len x (hidden_dim x 2)
            att_output = torch.sum(att_output, axis = 1)              # batch_size x (hidden_dim x 2)            
            att_output = att_output.unsqueeze(1)                      # batch_size x 1 x (hidden_dim x 2)
            x_case_dict[idx] = att_output                             # batch_size x 1 x (hidden_dim x 2)
        
        x_case_sent = torch.cat(list(x_case_dict.values()), dim = 1)  # batch_size x n_passages x (hidden_dim x 2)
               
        # Case document encoding
        self.lstm_case_doc.flatten_parameters()
        x_case_doc = self.lstm_case_doc(x_case_sent)                  # Tuple (len = 2)
        x_case_doc_fwd = x_case_doc[0][:, -1, 0:bilstm_b]             # batch_size x hidden_dim
        x_case_doc_bkwd = x_case_doc[0][:, 0, bilstm_b:bilstm_e]      # batch_size x hidden_dim
        x_case_doc = torch.cat((x_case_doc_fwd, x_case_doc_bkwd),
                               dim = 1)                               # batch_size x (hidden_dim x 2)
        x_case_doc = self.drops(x_case_doc)                           # batch_size x (hidden_dim x 2)
        
        # Concatenate article and passage encodings
        x = torch.cat((x_art, x_case_doc), dim = 1)                   # batch size x (hidden_dim x 2 x 2)
        x = self.fc_1(x)                                              # batch size x output_size
        
        # Sigmoid function
        x = self.sigmoid(x)                                           # batch size x output_size
        
        return x
