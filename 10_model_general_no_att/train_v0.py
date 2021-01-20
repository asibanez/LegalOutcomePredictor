# v0
# v1  -> Saves model parameters as json

#%% Imports

import os
import json
import torch
import pickle
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from model_v14 import ECHR_dataset, ECHR_model

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
        with torch.no_grad():
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

#%% Test function

def test_func(model, test_dl):
    model.eval()
    Y_predicted_score = []
    Y_predicted_binary = []
    Y_ground_truth = []

    for X_art, X_case, Y in test_dl:
        
        # Move to cuda
        if next(model.parameters()).is_cuda:
            X_art = X_art.to(device)
            X_case = X_case.to(device)
            Y = Y.to(device)
        
        # Compute predictions and append
        with torch.no_grad():
            pred_batch_score = model(X_art, X_case).view(-1)
            pred_batch_binary = torch.round(pred_batch_score.view(pred_batch_score.shape[0]))
            Y_predicted_score += pred_batch_score.tolist()
            Y_predicted_binary += pred_batch_binary.tolist()
            Y_ground_truth += Y.tolist()
        
    return Y_predicted_score, Y_predicted_binary, Y_ground_truth

#%% Path definition
"""
run_folder = os.path.join(os.path.split(os.getcwd())[0], '01_data', '02_runs', '20_art3_50p_art_dim_64_20_epochs') 
path_model_train = os.path.join(run_folder, 'model_train.pkl')
path_model_dev = os.path.join(run_folder, 'model_dev.pkl')
path_model_test = os.path.join(run_folder, 'model_test.pkl')
output_path_model = os.path.join(run_folder, 'model.pt')
output_path_results = os.path.join(run_folder, 'results.pkl')
output_path_params = os.path.join(run_folder, 'params.json')
input_path_id_2_embed = os.path.join(os.path.split(os.getcwd())[0], '01_data', '01_preprocessed', 'id_2_embed_dict.pkl')

"""
run_folder = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/01_data/02_runs/12_art_3_300_pass'
path_model_train = os.path.join(run_folder, 'model_train.pkl')
path_model_dev = os.path.join(run_folder, 'model_dev.pkl')
path_model_test = os.path.join(run_folder, 'model_test.pkl')
output_path_model = os.path.join(run_folder, 'model.pt')
output_path_results = os.path.join(run_folder, 'results.pkl')
output_path_params = os.path.join(run_folder, 'params.json')
input_path_id_2_embed = 'C://Users//siban//Dropbox//CSAIL//Projects//12_Legal_Outcome_Predictor//01_data/01_preprocessed//id_2_embed_dict.pkl'

#%% Global initialization

debug_flag = False
art_text = True
seq_len = 512
num_passages = 50

seed = 1234
n_epochs = 20
batch_size = 200
learning_rate = 0.001
dropout = 0.4
momentum = 0.9
wd = 0.00001

use_cuda = True
device = 'cuda:0'

embed_dim = 200
hidden_dim = 64
att_dim = 100
output_size = 1
pad_idx = 0

#%% Load data

print('Loading data')
with open(input_path_id_2_embed, 'rb') as fr:
    id_2_embed = pickle.load(fr)
model_train = pd.read_pickle(path_model_train)
model_dev = pd.read_pickle(path_model_dev)
model_test = pd.read_pickle(path_model_test)
print('Done')

#%% Slice data for debugging

if debug_flag == True:
    model_train = model_train[0:50]
    model_dev = model_dev[0:int(50 * 0.2)]
    model_test = model_test[0:int(50 * 0.2)]
    
#%% Instantiate dataclasses

print('Instantiating dataclases')
train_dataset = ECHR_dataset(model_train)
dev_dataset = ECHR_dataset(model_dev)
test_dataset = ECHR_dataset(model_test)
print('Done')

#%% Instantiate dataloaders

print('Instantiating dataloaders')
train_dl = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
dev_dl = DataLoader(dev_dataset, batch_size = batch_size * 2, shuffle = False)
test_dl = DataLoader(test_dataset, batch_size = batch_size * 2, shuffle = False)
print('Done')

#%% Instantiate model

pretrained_embeddings = torch.FloatTensor(list(id_2_embed.values()))
model = ECHR_model(embed_dim, hidden_dim, output_size, pretrained_embeddings,
                   att_dim, dropout, art_text, seq_len, num_passages)

# Move to cuda
if use_cuda and torch.cuda.is_available():
    print('Moving model to cuda')
    model = model.to(device)
    print('Done')

print(model)

#%% Instantiate optimizer & criterion

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate,
                             weight_decay = wd)
criterion = nn.BCELoss()

#%% Training

train_loss_history = []
val_loss_history = []

for epoch in tqdm(range(0, n_epochs), desc = 'Training'):
    train_loss, train_loss_history = train_epoch_func(model, criterion,
                                                      optimizer, train_dl, train_loss_history)
    print("training loss: ", train_loss)
    _, _, val_loss_history = val_epoch_func(model, criterion, dev_dl, val_loss_history)

#%% Testing

Y_predicted_score, Y_predicted_binary, Y_ground_truth = test_func(model, test_dl)

#%% Compute Metrics

tn, fp, fn, tp = confusion_matrix(Y_ground_truth, Y_predicted_binary).ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
auc = roc_auc_score(Y_ground_truth, Y_predicted_score)

# %% Print results

print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1: {f1:.3f}\n')
print(f'AUC: {auc:.3f}\n')

#%% Save model 

torch.save(model, output_path_model)

#%% Save results

results = {'training_loss': train_loss_history,
           'validation_loss': val_loss_history,
           'Y_test_ground_truth': Y_ground_truth,
           'Y_test_prediction_scores': Y_predicted_score,
           'Y_test_prediction_binary': Y_predicted_binary}
with open(output_path_results, 'wb') as fw:
    pickle.dump(results, fw) 

#%% Save model parameters

model_params = {'debug_flag': debug_flag,
                'art_text': art_text,
                'seq_len': seq_len,
                'num_passages': num_passages,              
                'seed': seed,
                'n_epochs': n_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'dropout': dropout,
                'momentum': momentum,
                'wd': wd,
                'use_cuda': use_cuda,
                'device': device,
                'embed_dim': embed_dim,
                'hidden_dim': hidden_dim,
                'att_dim': att_dim,
                'output_size': output_size,
                'pad_idx': pad_idx}

with open(output_path_params, 'w') as fw:
    json.dump(model_params, fw)

#%%