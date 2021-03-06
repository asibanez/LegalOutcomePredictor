# v0
# v1  -> Saves model parameters as json
# v2  -> Adds flexible multiprocessing
# v3  -> Train and validation accuracy histories added
# v4  -> Updated train and validation metrics history saving

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

def train_epoch_func(model, criterion, optimizer, train_dl):
    model.train()
    sum_correct = 0
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
        pred = torch.round(pred.view(pred.shape[0]))
        sum_correct += (pred == Y).sum().item()   
        
    avg_train_loss = sum_train_loss / total_entries
    train_accuracy = sum_correct / total_entries
    print(f'train loss: {avg_train_loss:.4f} and accuracy: {train_accuracy:.4f}')
    
    return avg_train_loss, train_accuracy

#%% Validation function

def val_epoch_func(model, criterion, dev_dl):
    model.eval()
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
    
    avg_val_loss = sum_val_loss / total_entries
    val_accuracy = sum_correct / total_entries
    print(f'valid loss: {avg_val_loss:.4f} and accuracy: {val_accuracy:.4f}')
    
    return avg_val_loss, val_accuracy

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

run_folder = os.path.join(os.path.split(os.getcwd())[0], '01_data/02_runs/05_art_05/45_test')
path_model_train = os.path.join(os.path.split(run_folder)[0], '00_input_data', 'model_train.pkl')
path_model_dev = os.path.join(os.path.split(run_folder)[0], '00_input_data', 'model_dev.pkl')
path_model_test = os.path.join(os.path.split(run_folder)[0], '00_input_data', 'model_test.pkl')
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
"""

#%% Global initialization

debug_flag = False
save_model_steps_flag = True
art_text = True
seq_len = 512
num_passages = 50

seed = 1234
n_epochs = 2
batch_size = 600
learning_rate = 1e-4
dropout = 0.4
momentum = 0.9
wd = 0.00001

use_cuda = True
gpu_ids = [3,4,5]

embed_dim = 200
hidden_dim = 100
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
                   dropout, art_text, seq_len, num_passages)

# Set device and move model to device
if use_cuda and torch.cuda.is_available():
    print('Moving model to cuda')
    if len(gpu_ids) > 1:
        device = torch.device(f'cuda:{gpu_ids[0]}')
        model = nn.DataParallel(model, device_ids = gpu_ids)
        model = model.to(device)
    else:
        device = torch.device('cuda:' + str(gpu_ids[0]))
        model = model.to(device)
    print('Done')
else:
    device = torch.device('cpu')
    model = model.to(device)

print(model)

#%% Instantiate optimizer & criterion

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate,
                             weight_decay = wd)
criterion = nn.BCELoss()

#%% Training

train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

for idx, epoch in enumerate(tqdm(range(0, n_epochs), desc = 'Training')):
    
    train_loss, train_acc = train_epoch_func(model, criterion, optimizer, train_dl)
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    
    val_loss, val_acc = val_epoch_func(model, criterion, dev_dl)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)
    
    if save_model_steps_flag == True:
        torch.save(model, output_path_model + '.' + str(idx))

#%% Testing

Y_predicted_score, Y_predicted_binary, Y_ground_truth = test_func(model, test_dl)

#%% Compute Metrics

tn, fp, fn, tp = confusion_matrix(Y_ground_truth, Y_predicted_binary).ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
auc = roc_auc_score(Y_ground_truth, Y_predicted_score)

#%% Print results

print(f'\nPrecision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1: {f1:.4f}')
print(f'AUC: {auc:.4f}\n')

#%% Save model 

torch.save(model, output_path_model)

#%% Save results

results = {'training_loss': train_loss_history,
           'training_acc': train_acc_history,
           'validation_loss': val_loss_history,
           'validation_acc': val_loss_history,
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
                'device': gpu_ids,
                'embed_dim': embed_dim,
                'hidden_dim': hidden_dim,
                'output_size': output_size,
                'pad_idx': pad_idx}

with open(output_path_params, 'w') as fw:
    json.dump(model_params, fw)

#%%
