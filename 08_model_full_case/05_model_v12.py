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

#%% Imports

import os
import torch
import pickle
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve

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
    
    def __init__(self, input_size, hidden_dim, ouput_size, pretrained_embeddings,
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
        self.fc_1 = nn.Linear(in_features = self.hidden_dim * 2 * 2,
                              out_features = self.output_size)
        
        # Sigmoid
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, X_art, X_case):
        # Embedding
        x_art = self.embed(X_art)                                      # batch_size x seq_len x embed_dim
        x_case = self.embed(X_case)                                    # batch_size x (seq_len x n_passages) x embed_dim
        
        # Article encoding
        x_art = self.lstm_art(x_art)                                   # Tuple (len = 2)
        x_art_fwd = x_art[0][:, -1, 0:64]                              # batch_size x hidden_dim
        x_art_bkwd = x_art[0][:, 0, 64:128]                            # batch_size x hidden_dim
        x_art = torch.cat((x_art_fwd, x_art_bkwd), dim = 1)            # batch_size x (hidden_dim x 2)
        x_art = self.drops(x_art)                                      # batch_size x (hidden_dim x 2) 
        
        # Case sentence encoding
        #x_case_sent = torch.FloatTensor()
        x_case_sent = torch.cuda.FloatTensor()
        
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
        x = torch.cat((x_art, x_case_doc), dim = 1)                   # batch size x (hidden_dim x 2 x 2)
        
        # Fully connected layer
        x = self.fc_1(x)                                              # batch size x output_size
        
        # Sigmoid function
        x = self.sigmoid(x)                                           # batch size x output_size
        
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

run_folder = os.path.join(os.path.split(os.getcwd())[0], '01_data', '02_runs', '13_art_6_50_pass') 
path_model_train = os.path.join(run_folder, 'model_train.pkl')
path_model_dev = os.path.join(run_folder, 'model_dev.pkl')
path_model_test = os.path.join(run_folder, 'model_test.pkl')
output_path_model = os.path.join(run_folder, 'model.pt')
output_path_results = os.path.join(run_folder, 'results.pkl')
input_path_id_2_embed = os.path.join(os.path.split(os.getcwd())[0], '01_data', '01_preprocessed', 'id_2_embed_dict.pkl')

<<<<<<< HEAD:08_model_full_case/05_model_v12.py
"""
run_folder = 'C://Users//siban//Dropbox/CSAIL//Projects//12_Legal_Outcome_Predictor//01_data//02_runs//07_art_3_no_att'
=======
run_folder = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/01_data/02_runs/12_art_6_300_pass'
>>>>>>> 3200aa970e2b6a6e6a7480ad04af2934c10de7be:08_model_full_case/05_model_v11.py
path_model_train = os.path.join(run_folder, 'model_train.pkl')
path_model_dev = os.path.join(run_folder, 'model_dev.pkl')
path_model_test = os.path.join(run_folder, 'model_test.pkl')
output_path_model = os.path.join(run_folder, 'model.pt')
output_path_results = os.path.join(run_folder, 'results.pkl')
input_path_id_2_embed = 'C://Users//siban//Dropbox//CSAIL//Projects//12_Legal_Outcome_Predictor//01_data/01_preprocessed//id_2_embed_dict.pkl'
"""

#%% Global initialization

seed = 1234
n_epochs = 20
batch_size = 500
learning_rate = 0.001
dropout = 0.4
momentum = 0.9
wd = 0.00001
use_cuda = True
device = torch.device('cuda:0')

embed_dim = 200
input_size = embed_dim
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

"""
#-------------------------------- FOR DEBUGGING -------
model_train = model_train[0:50]
model_dev = model_dev[0:int(50 * 0.2)]
model_test = model_test[0:int(50 * 0.2)]
#------------------------------------------------------
"""

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
model = ECHR_model(input_size, hidden_dim, output_size, pretrained_embeddings,
                   att_dim, dropout)

# Move to cuda
if use_cuda and torch.cuda.is_available():
    print('Moving model to cuda')
    model = model.to(device)
    print('Done')

print(model)

#%% Instantiate optimizer & criterion

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate,
                             weight_decay = wd)
#optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate,
#                            momentum = momentum)
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

print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1: {f1:.3f}\n')
print(f'AUC: {auc:.3f}\n')

#%% Save model and results

torch.save(model, output_path_model)
results = {'training_loss': train_loss_history,
           'validation_loss': val_loss_history,
           'Y_test_ground_truth': Y_ground_truth,
           'Y_test_prediction_scores': Y_predicted_score,
           'Y_test_prediction_binary': Y_predicted_binary}
with open(output_path_results, 'wb') as fw:
    pickle.dump(results, fw) 
