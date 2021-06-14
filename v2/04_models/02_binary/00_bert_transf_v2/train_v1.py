# v0
# v1 -> Saves full checkpoints

#%% Imports

import os
import json
import random
import argparse
import datetime
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_v0 import ECHR2_dataset, ECHR2_model

#%% Train function

def train_epoch_f(args, epoch, model, criterion, 
                  optimizer, train_dl,
                  output_train_log_file_path, device):
   
    model.train()
    sum_correct = 0
    total_entries = 0
    sum_train_loss = 0
    
    for step_idx, (X_facts_ids, X_facts_token_types, X_facts_attn_masks,
                   X_echr_ids, X_echr_token_types, X_echr_attn_masks, Y_labels) in \
        tqdm(enumerate(train_dl), total = len(train_dl), desc = 'Training epoch'):
        
        # Move data to cuda
        if next(model.parameters()).is_cuda:
            X_facts_ids = X_facts_ids.to(device)
            #X_facts_token_types = X_facts_token_types.to(device)
            X_facts_attn_masks = X_facts_attn_masks.to(device)
            X_echr_ids = X_echr_ids.to(device)
            #X_echr_token_types = X_echr_token_types.to(device)
            X_echr_attn_masks = X_echr_attn_masks.to(device)
            Y_labels = Y_labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        #Forward + backward + optimize
        pred = model(X_facts_ids, X_facts_token_types, X_facts_attn_masks,
                     X_echr_ids, X_echr_token_types, X_echr_attn_masks)
        pred = pred.view(-1)
        loss = criterion(pred, Y_labels)
        
        # Backpropagate
        loss.backward()
        
        # Update model
        optimizer.step()           
        
        # Book-keeping
        current_batch_size = X_facts_ids.size()[0]
        total_entries += current_batch_size
        sum_train_loss += (loss.item() * current_batch_size)
        pred = torch.round(pred)
        sum_correct += (pred == Y_labels).sum().item()
      
        # Write log file
        with open(output_train_log_file_path, 'a+') as fw:
            fw.write(f'{str(datetime.datetime.now())} Epoch {epoch + 1} of {args.n_epochs}' +
                     f' Step {step_idx + 1:,} of {len(train_dl):,}\n')

    # Compute metrics
    avg_train_loss = sum_train_loss / total_entries
    avg_train_acc = sum_correct / total_entries
    print(f'\nTrain loss: {avg_train_loss:.4f} and accuracy: {avg_train_acc:.4f}')

    # Write log file
    with open(output_train_log_file_path, 'a+') as fw:
        fw.write(f'{str(datetime.datetime.now())} Train loss: {avg_train_loss:.4f} and ' +
                 f'accuracy: {avg_train_acc:.4f}\n')
    
    return avg_train_loss, avg_train_acc

#%% Validation function

def val_epoch_f(args, model, criterion, dev_dl, device):
    model.eval()
    sum_correct = 0
    sum_val_loss = 0
    total_entries = 0

    for X_facts_ids, X_facts_token_types, X_facts_attn_masks,\
        X_echr_ids, X_echr_token_types, X_echr_attn_masks, Y_labels in \
        tqdm(dev_dl, desc = 'Validation'):
        
        # Move to cuda
        if next(model.parameters()).is_cuda:
            X_facts_ids = X_facts_ids.to(device)
            #X_facts_token_types = X_facts_token_types.to(device)
            X_facts_attn_masks = X_facts_attn_masks.to(device)
            X_echr_ids = X_echr_ids.to(device)
            #X_echr_token_types = X_echr_token_types.to(device)
            X_echr_attn_masks = X_echr_attn_masks.to(device)
            Y_labels = Y_labels.to(device)
                    
        # Compute predictions:
        with torch.no_grad():
            pred = model(X_facts_ids, X_facts_token_types, X_facts_attn_masks,
                     X_echr_ids, X_echr_token_types, X_echr_attn_masks)
            pred = pred.view(-1)
            loss = criterion(pred, Y_labels)
        
        # Book-keeping
        current_batch_size = X_facts_ids.size()[0]
        total_entries += current_batch_size
        sum_val_loss += (loss.item() * current_batch_size)
        pred = torch.round(pred)
        sum_correct += (pred == Y_labels).sum().item()

    avg_val_loss = sum_val_loss / total_entries
    avg_val_accuracy = sum_correct / total_entries
    print(f'\n\tvalid loss: {avg_val_loss:.4f} and accuracy: {avg_val_accuracy:.4f}')
    
    return avg_val_loss, avg_val_accuracy

#%% Main

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default = None, type = str, required = True,
                       help = 'input folder')
    parser.add_argument('--output_dir', default = None, type = str, required = True,
                       help = 'output folder')
    parser.add_argument('--n_epochs', default = None, type = int, required = True,
                       help = 'number of total epochs to run')
    parser.add_argument('--batch_size', default = None, type = int, required = True,
                       help = 'train batch size')
    parser.add_argument('--shuffle_train', default = None, type = str, required = True,
                       help = 'shuffle train set')
    parser.add_argument('--train_toy_data', default = None, type = str, required = True,
                       help = 'Use toy dataset for training')
    parser.add_argument('--len_train_toy_data', default = None, type = int, required = True,
                       help = 'train toy data size')
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
    parser.add_argument('--num_labels', default = None, type = int, required = True,
                       help = 'number of labels')
    parser.add_argument('--n_heads', default = None, type = int, required = True,
                       help = 'number of transformer heads')
    parser.add_argument('--hidden_dim', default = None, type = int, required = True,
                       help = 'lstm hidden dimension')
    parser.add_argument('--max_n_pars_facts', default = None, type = int, required = True,
                       help = 'max number of paragrpahs in facts')
    parser.add_argument('--max_n_pars_echr', default = None, type = int, required = True,
                       help = 'max number of paragrahs in echr articles')
    parser.add_argument('--pad_idx', default = None, type = int, required = True,
                       help = 'pad token index')  
    parser.add_argument('--save_final_model', default = None, type = str, required = True,
                       help = 'final .pt model is saved in output folder')
    parser.add_argument('--save_model_steps', default = None, type = str, required = True,
                       help = 'intermediate .pt models saved in output folder')
    parser.add_argument('--save_step_cliff', default = None, type = int, required = True,
                       help = 'start saving models after cliff')
    parser.add_argument('--use_cuda', default = None, type = str, required = True,
                        help = 'use CUDA')
    parser.add_argument('--gpu_ids', default = None, type = str, required = True,
                        help='gpu IDs')
    args = parser.parse_args()
       
    # Path initialization train-dev
    path_model_train = os.path.join(args.input_dir, 'model_train.pkl')
    path_model_dev = os.path.join(args.input_dir, 'model_dev.pkl')
    
    # Path initialization output files
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
    print('Loading data')
    model_train = pd.read_pickle(path_model_train)
    model_dev = pd.read_pickle(path_model_dev)
    print('Done')
    
    if eval(args.train_toy_data) == True:
        model_train = model_train[0:args.len_train_toy_data]
        model_dev = model_dev[0:args.len_train_toy_data]
   
    # Instantiate dataclasses
    train_dataset = ECHR2_dataset(model_train)
    dev_dataset = ECHR2_dataset(model_dev)

    # Instantiate dataloaders
    train_dl = DataLoader(train_dataset, batch_size = args.batch_size,
                          shuffle = eval(args.shuffle_train), drop_last = True)
    dev_dl = DataLoader(dev_dataset, batch_size = args.batch_size * 2,
                        shuffle = False)

    # Instantiate model
    model = ECHR2_model(args)

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

    #print(model)

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

        val_loss, val_acc = val_epoch_f(args, model, criterion, dev_dl, device)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc) 

        # Save checkpoint
        if eval(args.save_model_steps) == True and epoch >= args.save_step_cliff:
            if len(args.gpu_ids) > 1 and eval(args.use_cuda) == True:
                model_state_dict_save = model.module.state_dict()
            else:
                model_state_dict_save = model.state_dict()
            torch.save({'epoch': epoch,
                        'model_state_dict': model_state_dict_save,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss},
                        output_path_model + '.' + str(epoch))

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
    model_params = {'input_dir': args.input_dir,
                    'n_epochs': args.n_epochs,
                    'batch_size': args.batch_size,
                    'shuffle_train': args.shuffle_train, 
                    'train_toy_data': args.train_toy_data,
                    'len_train_toy_data': args.len_train_toy_data,
                    'learning_rate': args.lr,
                    'wd': args.wd,
                    'dropout': args.dropout,
                    'momentum': args.momentum,
                    'seed': args.seed,
                    'seq_len': args.seq_len,
                    'num_labels': args.num_labels,              
                    'hidden_dim': args.hidden_dim,
                    'max_n_pars_facts': args.max_n_pars_facts,
                    'max_n_pars_echr': args.max_n_pars_echr,
                    'pad_idx': args.pad_idx,
                    'save_final_model': args.save_final_model,
                    'save_model_steps': args.save_model_steps,
                    'save_step_cliff': args.save_step_cliff,
                    'use_cuda': args.use_cuda,
                    'gpu_ids': args.gpu_ids}
    
    with open(output_path_params, 'w') as fw:
        json.dump(model_params, fw)
    
if __name__ == "__main__":
    main()
