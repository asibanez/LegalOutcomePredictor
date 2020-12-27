# v1 -> Pickle dataframes read with pandas - "Load data" section
# v2 -> Splits LSTM into 2 (LSTM and case)
# v3 -> Uses FastText embeddings
# v4 -> Conversion to token IDs, prunning and padding moved to preprocessing

#%% Imports

import os
import torch
import pickle
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Function definition
#%% DataClass

class ECHR_dataset(Dataset):
    def __init__(self, data_df):
        self.article_tensor = torch.LongTensor(data_df['article_text'].to_list())
        self.cases_tensor = torch.LongTensor(data_df['case_texts'].to_list())
        self.outcome_tensor = torch.LongTensor(data_df['outcome'].to_list())
        
    def __len__(self):
        return self.outcome_tensor.size()[0]
        
    def __getitem__(self, idx):
        X_article = self.article_tensor[idx, :]
        X_cases = self.cases_tensor[idx, :]
        Y = self.outcome_tensor[idx]
        
        return X_article, X_cases, Y

#%% Model

class ECHR_model(nn.Module):
    
    def __init__(self, input_size, hidden_size, ouput_size, pretrained_embeddings):
        super(ECHR_model, self).__init__()

        # Embedding
        self.embed = nn.Embedding.from_pretrained(pretrained_embeddings)
        
        # Encode article
        self.lstm_art = nn.LSTM(input_size = input_size,
                                hidden_size = hidden_size,
                                bidirectional = True,
                                batch_first = True)      
        
        # Encode cases
        self.lstm_case_1 = nn.LSTM(input_size = input_size,
                                 hidden_size = hidden_size,
                                 bidirectional = True,
                                 batch_first = True)
        
        self.lstm_case_2 = nn.LSTM(input_size = input_size,
                                 hidden_size = hidden_size,
                                 bidirectional = True,
                                 batch_first = True)      
        
        # Concatenate
        self.fc_1 = nn.Linear(in_features = 6 * hidden_size,
                              out_features = 1)
        
    def forward(self, X_art, X_case):
        # Embedding
        x_art = self.embed(X_art)
        x_case = self.embed(X_case)
        
        # LSTM article
        x_art = self.lstm_art(x_art)
        x_art_fwd = x_art[0][:, -1, 0:64]
        x_art_bkwd = x_art[0][:, 0, 64:128]
        x_art = torch.cat((x_art_fwd, x_art_bkwd), dim = 1)
        
        # LSTM cases
        x_case_1 = self.lstm_case_1(x_case[:, 0:512, :])
        x_case_1_fwd = x_case_1[0][:, -1, 0:64]
        x_case_1_bkwd = x_case_1[0][:, 0, 64:128]
        x_case_1 = torch.cat((x_case_1_fwd, x_case_1_bkwd), dim = 1)
        
        x_case_REST = self.lstm_case_1(x_case[:, 512:, :])
        x_case_REST_fwd = x_case_REST[0][:, -1, 0:64]
        x_case_REST_bkwd = x_case_REST[0][:, 0, 64:128]
        x_case_REST = torch.cat((x_case_REST_fwd, x_case_REST_bkwd), dim = 1)
        
        # Concatenate article & passages
        x = torch.cat((x_art, x_case_1, x_case_REST), dim = 1)
        x = self.fc_1(x)
        
        return x

#%% Train function

def train_epoch_func(model, criterion, optimizer, train_dl, train_loss_history):
    model.train()
    train_acc = 0
    total_entries = 0
    sum_train_loss = 0
    
    for X_art, X_case, Y in tqdm(train_dl, total = len(train_dl)):
        
        # Move to cuda
        if next(model.parameters()).is_cuda:
            X_art = X_art.to(device)
            X_case = X_case.to(device)
            Y = Y.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        #Forward + backward + optimize
        pred = model(X_art, X_case).view(-1)
        loss = criterion(pred, Y)
        
        # Backpropagate
        loss.backward()
        
        # Update model
        optimizer.step()           
        
        # Book-keeping
        current_batch_size = X_art.size()[0]
        total_entries += current_batch_size
        sum_train_loss += (loss.item() * current_batch_size)
        
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

    for X_art, X_case, Y in dev_dl:
        
        # Move to cuda
        if next(model.parameters()).is_cuda:
            X_art = X_art.to(device)
            X_case = X_case.to(device)
            Y = Y.to(device)
        
        # Compute predictions:
        pred = model(X_art, X_case).view(-1)
        loss = criterion(pred, Y)
        
        # Book-keeping
        current_batch_size = X_art.size()[0]
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
input_path_id_2_embed = os.path.join(base_folder, 'id_2_embed_dict.pkl')

#%% Variable initialization

n_epochs = 20
seed = 1234
max_seq_len = 512
batch_size = 5
embed_dim = 200
input_size = embed_dim
hidden_size = 64
output_size = 1
learning_rate = 0.001
momentum = 0.9
wd = 0.00001
use_cuda = True
pad_idx = 0
device = torch.device('cuda:3')

#%% Load data

print('Loading data')
with open(input_path_id_2_embed, 'rb') as fr:
    id_2_embed = pickle.load(fr)

model_train = pd.read_pickle(path_model_train)
model_dev = pd.read_pickle(path_model_dev)
model_test = pd.read_pickle(path_model_test)
print('Done')

#%% Instantiate dataclasses

print('Instantiating dataclases')
train_dataset = ECHR_dataset(model_train)
dev_dataset = ECHR_dataset(model_dev)
test_dataset = ECHR_dataset(model_test)
print('Done')

#%% Instantiate dataloaders

print('Instantiating dataloaders')
train_dl = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
dev_dl = DataLoader(dev_dataset, batch_size = batch_size, shuffle = True)
test_dl = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
print('Done')

#%% Instantiate model

pretrained_embeddings = torch.FloatTensor(list(id_2_embed.values()))
model = ECHR_model(input_size, hidden_size, output_size, pretrained_embeddings)

# Move to cuda
if use_cuda and torch.cuda.is_available():
    print('moving model to cuda')
    #model = model.cuda()
    model = model.to(device)

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
