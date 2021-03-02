#%% Imports

import os
import json
import random
import pickle
import argparse
import datetime
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from base.model_base_v16 import ECHR_dataset, ECHR_model

#%% Train function

def train_epoch_f(args, epoch, model, criterion, 
                  optimizer, train_dl,
                  output_train_log_file_path, device):
   
    model.train()
    sum_correct = 0
    total_entries = 0
    sum_train_loss = 0
    
    for idx, (X_art, X_case, Y) in tqdm(enumerate(train_dl),
                                        total = len(train_dl),
                                        desc = 'Training epoch'):
        
        # Move data to cuda
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
        
        # Write log file
        with open(output_train_log_file_path, 'a+') as fw:
            fw.write(f'{str(datetime.datetime.now())} Epoch {epoch + 1} of {args.n_epochs}' +
                     f' Step {idx + 1:,} of {len(train_dl):,}\n')

    # Compute metrics
    avg_train_loss = sum_train_loss / total_entries
    avt_train_acc = sum_correct / total_entries
    print(f'\nTrain loss: {avg_train_loss:.4f} and accuracy: {avt_train_acc:.4f}')

    # Write log file
    with open(output_train_log_file_path, 'a+') as fw:
        fw.write(f'{str(datetime.datetime.now())} Train loss: {avg_train_loss:.4f} and ' +
                 f'accuracy: {avt_train_acc:.4f}\n')
    
    return avg_train_loss, avt_train_acc

#%% Validation function

def val_epoch_f(model, criterion, dev_dl, device):
    model.eval()
    sum_correct = 0
    sum_val_loss = 0
    total_entries = 0

    for X_art, X_case, Y in tqdm(dev_dl, desc = 'Validation'):
        
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
    print(f'\n\tvalid loss: {avg_val_loss:.4f} and accuracy: {val_accuracy:.4f}')
    
    return avg_val_loss, val_accuracy

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default = None, type = str, required = True,
                       help = 'input folder')
    parser.add_argument('--output_dir', default = None, type = str, required = True,
                       help = 'output folder')
    parser.add_argument('--path_embed', default = None, type = str, required = True,
                       help = 'path to file with embeddings')
    parser.add_argument('--n_epochs', default = None, type = int, required = True,
                       help = 'number of total epochs to run')
    parser.add_argument('--batch_size', default = None, type = int, required = True,
                       help = 'train batch size')
    parser.add_argument('--lr', default = None, type = float, required = True,
                       help = 'learning rate')
    parser.add_argument('--wd', default = None, type = float, required = True,
                        help = 'weight decay')
    parser.add_argument('--dropout', default = None, type = float, required = True,
                       help = 'dropout')
    parser.add_argument('--momentum', default = None, type = float, required = True,
                       help = 'momentum')
    parser.add_argument('--seed', default = None, type = int, required = True,
                       help = 'random seed')
    parser.add_argument('--seq_len', default = None, type = int, required = True,
                       help = 'text sequence length')
    parser.add_argument('--num_passages', default = None, type = int, required = True,
                       help = 'number of leaf nodes considered')
    parser.add_argument('--embed_dim', default = None, type = int, required = True,
                       help = 'embedding dimension')
    parser.add_argument('--hidden_dim', default = None, type = int, required = True,
                       help = 'lstm hidden dimension')
    parser.add_argument('--att_dim', default = None, type = int, required = True,
                       help = 'attention layer dimension')
    parser.add_argument('--pad_idx', default = None, type = int, required = True,
                       help = 'pad token index')  
    parser.add_argument('--save_final_model', default = None, type = str, required = True,
                       help = 'final .pt model is saved in output folder')
    parser.add_argument('--save_model_steps', default = None, type = str, required = True,
                       help = 'intermediate .pt models saved in output folder')
    parser.add_argument('--use_cuda', default = None, type = str, required = True,
                        help = 'use CUDA')
    parser.add_argument('--gpu_ids', default = None, type = str, required = True,
                        help='gpu IDs')
    args = parser.parse_args()
    
    # Path initialization
    # Train Dev
    path_model_train = os.path.join(args.input_dir, 'model_train.pkl')
    path_model_dev = os.path.join(args.input_dir, 'model_dev.pkl')
    
    # Output files
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        print("Created folder : ", args.output_dir)
    output_train_log_file_path = os.path.join(args.output_dir, 'train_log.txt')
    output_path_model = os.path.join(args.output_dir, 'model.pt')
    output_path_results = os.path.join(args.output_dir, 'train_results.json')
    output_path_params = os.path.join(args.output_dir, 'params.json')
      
    # Global and seed initialization
    args.gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    random.seed = args.seed
    _ = torch.manual_seed(args.seed)

    # Train dev test sets load
    print(datetime.datetime.now(), 'Loading train data')
    model_train = pd.read_pickle(path_model_train)
    print(datetime.datetime.now(), 'Loading dev data')
    model_dev = pd.read_pickle(path_model_dev)
    print(datetime.datetime.now(), 'Done')

#### Slicing for debugging
#    model_train = model_train[0:50]
#    model_dev = model_dev[0:10]
####

    # Load embeddings
    print(datetime.datetime.now(), 'Loading embeddings')
    with open(args.path_embed, 'rb') as fr:
        id_2_embed = pickle.load(fr)
    print(datetime.datetime.now(), 'Done')
    
    # Instantiate dataclasses
    train_dataset = ECHR_dataset(model_train)
    dev_dataset = ECHR_dataset(model_dev)

    # Instantiate dataloaders
    train_dl = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True,
                         drop_last = True)
    dev_dl = DataLoader(dev_dataset, batch_size = args.batch_size * 2, shuffle = False)

    # Instantiate model
    pretrained_embeddings = torch.FloatTensor(list(id_2_embed.values()))
    model = ECHR_model(args, pretrained_embeddings)

    # Set device and move model to device
    if eval(args.use_cuda) and torch.cuda.is_available():
        print('Moving model to cuda')
        if len(args.gpu_ids) > 1:
            device = torch.device('cuda', args.gpu_ids[0])
            model = nn.DataParallel(model, device_ids = args.gpu_ids)
            model = model.cuda(device)
        else:
            device = torch.device('cuda', args.gpu_ids[0])
            model = model.cuda(device)
        print('Done')
    else:
        device = torch.device('cpu')
        model = model.to(device)

    print(model)

    # Instantiate optimizer & criterion
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = args.lr,
                                 weight_decay = args.wd)
    criterion = nn.BCELoss()

    # Training procedure
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    start_time = datetime.datetime.now()
    
    for epoch in tqdm(range(args.n_epochs), desc = 'Training dataset'):
    
        train_loss, train_acc = train_epoch_f(args, epoch, model, criterion,
                                              optimizer, train_dl,
                                              output_train_log_file_path, device)
                
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        val_loss, val_acc = val_epoch_f(model, criterion, dev_dl, device)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc) 

        if eval(args.save_model_steps) == True:
            if len(args.gpu_ids) > 1 and eval(args.use_cuda) == True:
                torch.save(model.module.state_dict(), output_path_model + '.' + str(epoch))
            else:
                torch.save(model.state_dict(), output_path_model + '.' + str(epoch))

    end_time = datetime.datetime.now()
            
    # Save model
    if eval(args.save_final_model) == True:
        if len(args.gpu_ids) > 1 and eval(args.use_cuda) == True:
            torch.save(model.module.state_dict(), output_path_model)
        else:
            torch.save(model.state_dict(), output_path_model)
    
    # Save results
    results = {'training_loss': train_loss_history,
               'training_acc': train_acc_history,
               'validation_loss': val_loss_history,
               'validation_acc': val_loss_history,
               'start time': str(start_time),
               'end time': str(end_time)}
    with open(output_path_results, 'w') as fw:
        json.dump(results, fw)
    
    # Save model parameters
    model_params = {'n_epochs': args.n_epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.lr,
                    'wd': args.wd,
                    'dropout': args.dropout,
                    'momentum': args.momentum,
                    'seed': args.seed,
                    'seq_len': args.seq_len,
                    'num_passages': args.num_passages,              
                    'embed_dim': args.embed_dim,
                    'hidden_dim': args.hidden_dim,
                    'att_dim': args.att_dim,
                    'pad_idx': args.pad_idx,
                    'save_final_model': args.save_final_model,
                    'save_model_steps': args.save_model_steps,
                    'use_cuda': args.use_cuda,
                    'gpu_ids': args.gpu_ids}
    
    with open(output_path_params, 'w') as fw:
        json.dump(model_params, fw)
    
if __name__ == "__main__":
    main()
