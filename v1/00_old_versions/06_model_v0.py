#%% Imports
import os
import pickle
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Function definition
#%% DataClass
class ECHR_dataset(Dataset):
    def __init__(self, data_df, tok_to_id, id_2_embed):
        self.data_df = data_df
        
    def __len__(self):
        return len(self.data_df)
        
    def __getitem__(self, idx):
        item = self.data_df.iloc[idx]
        
        outcome = item['outcome']
        
        # Get art_text_tokens and case_text_tokens from dataframe
        art_text_tokens = item['article_text']
        case_texts_tokens = item['case_texts']
        
        # Prune sequences to max seq len
        art_text_tokens = art_text_tokens[:max_seq_len]
        case_texts_tokens = [x[:max_seq_len] for x in case_texts_tokens]
        
        # Pad sequences to max seq len
        art_text_tokens = art_text_tokens + ['<pad>'] * (max_seq_len - len(art_text_tokens))
        case_texts_tokens = [x + ['<pad>'] * (max_seq_len - len(x)) for x in case_texts_tokens]
        
        # Flatten case_texts_tokens
        case_text_tokens_1D = [x for sublist in case_texts_tokens for x in sublist]
                
        # Compute tokens -> ids
        art_text_ids = []
        case_text_ids = []
        
        for token in art_text_tokens:
            if token in list(tok_to_id.keys()):
                art_text_ids.append(tok_to_id[token])
            else:
                art_text_ids.append(tok_to_id['<UNK>'])
                
        for token in case_text_tokens_1D:
            if token in list(tok_to_id.keys()):
                case_text_ids.append(tok_to_id[token])
            else:
                case_text_ids.append(tok_to_id['<UNK>'])
           
        # Compute ids -> embeddings
        #art_text_emb = [id_2_embed[x] for x in art_text_ids]
        #case_text_emb = [id_2_embed[x] for x in case_text_ids]
        
        # Convert to numpy arrays
        art_text_ids = np.array(art_text_ids)
        case_text_ids = np.array(case_text_ids)
        
        # Convert to tensors and concatenate X_data
        art_text_ids = torch.tensor(art_text_ids)
        case_text_ids = torch.tensor(case_text_ids)
        X = torch.cat((art_text_ids, case_text_ids), dim=0)
        Y = torch.tensor(outcome).float()
        
        return X, Y

#%% Model

class ECHR_model(nn.Module):
    
    def __init__(self, vocab_size, embed_dim, pad_idx, input_size, hidden_size, ouput_size):
        super(ECHR_model, self).__init__()
        self.embed = nn.Embedding(num_embeddings = vocab_size,
                                  embedding_dim = embed_dim)
        #self.embed.from_pretrained(ARRAY CON IDS)
        self.lstm = nn.LSTM(input_size = input_size,
                                hidden_size = hidden_size,
                                bidirectional = True,
                                batch_first = True)      
        self.fc_1 = nn.Linear(in_features = 2 * hidden_size,
                              out_features = 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        x = self.embed(input.long())
        x = self.lstm(x)
        x_fwd = x[0][:, -1, 0:64]
        x_bcwd = x[0][:, 0, 64:128]
        x = torch.cat((x_fwd, x_bcwd), dim = 1)
        x = self.fc_1(x)
        return x

#%% Train function

def train_epoch_func(model, criterion, optimizer, train_dl, train_loss_history):
    model.train()
    train_acc = 0
    total_entries = 0
    sum_train_loss = 0
    
    for X, Y in tqdm(train_dl, total = len(train_dl)):
        
        # Move to cuda
        if next(model.parameters()).is_cuda:
            X, Y = X.cuda(), Y.cuda()
        
        # Zero gradients
        optimizer.zero_grad()
        
        #Forward + backward + optimize
        pred = model(X).view(-1)
        loss = criterion(pred, Y)
        
        # Backpropagate
        loss.backward()
        
        # Update model
        optimizer.step()           
        
        # Book-keeping
        current_batch_size = X.size()[0]
        total_entries += current_batch_size
        sum_train_loss += (loss.item() * current_batch_size)
        
        #print("loss = ", loss.item())

    avg_train_loss = sum_train_loss / total_entries
    train_loss_history.append(avg_train_loss)
    
    return avg_train_loss, train_loss_history

#%% Validation function

def val_epoch_func(model, criterion, dev_dl, val_loss_history):
    model.eval()
    val_acc = 0
    sum_correct = 0
    sum_val_loss = 0
    total_entries = 0

    for X, Y in dev_dl:
        
        # Move to cuda
        if next(model.parameters()).is_cuda:
            X, Y = X.cuda(), Y.cuda()
        
        # Compute predictions:
        pred = model(X).view(-1)
        loss = criterion(pred, Y)
        
        # Book-keeping
        current_batch_size = X.size()[0]
        total_entries += current_batch_size
        sum_val_loss += (loss.item() * current_batch_size)
        pred = torch.round(pred.view(pred.shape[0]))
        sum_correct += (pred == Y).sum().item()             
    
    avg_loss = sum_val_loss / total_entries
    accuracy = sum_correct / total_entries
    val_loss_history.append(avg_loss)
    print("valid loss %.3f and accuracy %.3f" % (avg_loss, accuracy))
    
    return avg_loss, accuracy, val_loss_history

#%% Path definition

base_folder = os.path.join(os.getcwd(),'01_data', '01_preprocessed') 
path_model_train = os.path.join(base_folder, 'model_train.pkl')
path_model_dev = os.path.join(base_folder, 'model_dev.pkl')
path_model_test = os.path.join(base_folder, 'model_test.pkl')
input_path_tok_2_id = os.path.join(base_folder, 'tok_2_id')
input_path_id_2_embed = os.path.join(base_folder, 'id_2_embed')

#%% Variable initialization

n_epochs = 5
seed = 1234
max_seq_len = 512
batch_size = 20
embed_dim = 128
input_size = embed_dim
hidden_size = 64
output_size = 1
learning_rate = 0.001
momentum = 0.9
wd = 0.00001
use_cuda = True

#%% Load data

with open(input_path_tok_2_id, 'rb') as fr:
    tok_to_id = pickle.load(fr) 

with open(input_path_id_2_embed, 'rb') as fr:
    id_2_embed = pickle.load(fr)

#with open(path_model_train, 'rb') as fr:
#    model_train = pickle.load(fr)

#with open(path_model_dev, 'rb') as fr:
#    model_dev = pickle.load(fr)

#with open(path_model_test, 'rb') as fr:
#    model_test = pickle.load(fr)

model_train = pd.read_pickle(path_model_train)
model_dev = pd.read_pickle(path_model_dev)
model_test = pd.read_pickle(path_model_test)


#%% Instantiate dataclasses

train_dataset = ECHR_dataset(model_train, tok_to_id, id_2_embed)
dev_dataset = ECHR_dataset(model_dev, tok_to_id, id_2_embed)
test_dataset = ECHR_dataset(model_test, tok_to_id, id_2_embed)

#%% Instantiate dataloaders

train_dl = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
dev_dl = DataLoader(dev_dataset, batch_size = batch_size, shuffle = True)
test_dl = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

#%% Instantiate model

vocab_size = len(id_2_embed)
seq_len = 512 * 6
model = ECHR_model(vocab_size=vocab_size, embed_dim=embed_dim,
                   pad_idx=0, input_size=input_size,
                   hidden_size=hidden_size, ouput_size=output_size)

# Move to cuda
if use_cuda and torch.cuda.is_available():
    print('moving model to cuda')
    model = model.cuda()

print(model)

#%% Instantiate optimizer & criterion

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
criterion = nn.BCEWithLogitsLoss()

#%% Training

train_loss_history = []
val_loss_history = []

for epoch in tqdm(range(0, n_epochs), desc = 'Training'):
    train_loss, train_loss_history = train_epoch_func(model, criterion,
                                                      optimizer, train_dl, train_loss_history)
    print("training loss: ", train_loss)
    _, _, val_loss_history = val_epoch_func(model, criterion, dev_dl, val_loss_history)

#%%
