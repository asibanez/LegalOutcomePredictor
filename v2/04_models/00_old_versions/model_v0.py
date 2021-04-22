# Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
#import inspect
#lines = inspect.getsource(AutoModelForSequenceClassification)

#%% DataClass definition
class ECHR2_dataset(Dataset):
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
class ECHR2_model(nn.Module):
            
    def __init__(self, args):
        super(ECHR2_model, self).__init__()

        self.num_pars = 37
        self.h_dim = 512
        self.n_heads = 8
        self.n_labels = 10
        self.seq_len = 512
        self.dropout = 0.4 #args.dropout
        
             
        # Bert model
        self.model_name = 'nlpaueb/legal-bert-small-uncased'
        self.bert_model = AutoModel.from_pretrained(self.model_name)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                            use_fast = True)
        
        # Transformer layer
        self.transf_enc = nn.TransformerEncoderLayer(d_model = self.h_dim,
                                                     nhead = self.n_heads)
    
        # Fully connected output
        self.fc_out = nn.Linear(in_features = self.h_dim,
                                out_features = self.n_labels)

        # Pooling
        self.max_pool = nn.MaxPool1d(kernel_size = self.num_pars)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

        # Dropout
        self.drops = nn.Dropout(self.dropout)
            
    def forward(self, facts):
        bert_out = {}
        for idx, fact in enumerate(facts):
            tokens = self.bert_tokenizer(fact,
                                         return_tensors = 'pt',
                                         padding = 'max_length',
                                         truncation = True,
                                         max_length = self.seq_len)

            output = self.bert_model(**tokens,output_hidden_states = True)
            #bert_out[idx] = output['last_hidden_state']
            bert_out[idx] = output['pooler_output']
            
        aux = torch.cat(list(bert_out.values()),dim=0)      # n_pars x h_dim
    
        x = aux.unsqueeze(0)                                # batch_size x n_pars x h_dim
        x = self.transf_enc(x)                              # batch_size x n_pars x h_dim
        x = x.transpose(1,2)                                # batch_size x h_dim x n_pars
        x = self.max_pool(x)                                # batch_size x h_dim x 1
        x = x.squeeze(2)                                    # batch_size x h_dim
        x = self.fc_out(x)                                  # batch_size x n_lab
        x = self.sigmoid(x)                                 # batch_size x n_lab

        return x

#%% Load data
test_set = load_dataset('ecthr_cases', split = 'test')
facts = test_set[0]['facts']

#%% Instantiate model and tokenizer
ECHR2_model = ECHR2_model(None)

#%% Compute predictions
pred = ECHR2_model(facts)