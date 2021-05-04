# v0

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
from model_v4 import ECHR2_dataset, ECHR2_model

#%% Train function
def train_epoch_f(args, epoch, model, criterion, 
                  optimizer, train_dl,
                  output_train_log_file_path, device):
   
    model.train()
    sum_correct = {x:0 for x in range (0,args.num_labels)}
    total_entries = 0
    sum_train_loss = 0
    
    for step_idx, (X_bert_encoding, X_transf_mask, Y_labels) in \
        tqdm(enumerate(train_dl), desc = 'Training epoch'):
        
        # Move data to cuda
        if next(model.parameters()).is_cuda:
            X_bert_encoding = X_bert_encoding.to(device)
            X_transf_mask = X_transf_mask.to(device)
            Y_labels = Y_labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        #Forward + backward + optimize
        pred = model(X_bert_encoding, X_transf_mask)
        loss = criterion(pred, Y_labels)
        
        # Backpropagate
        loss.backward()
        
        # Update model
        optimizer.step()           
        
        # Book-keeping
        current_batch_size = X_bert_encoding.size()[0]
        total_entries += current_batch_size
        sum_train_loss += (loss.item() * current_batch_size)
        pred = torch.round(pred)
        for idx in range (0, args.num_labels):
            pred_1_label = pred[:, idx]
            Y_1_label = Y_labels[:, idx]
            sum_correct[idx] = (pred_1_label == Y_1_label).sum().item()
      
        # Write log file
        with open(output_train_log_file_path, 'a+') as fw:
            fw.write(f'{str(datetime.datetime.now())} Epoch {epoch + 1} of {args.n_epochs}' +
                     f' Step {step_idx + 1:,} of {len(train_dl):,}\n')

    # Compute metrics
    avg_train_loss = sum_train_loss / total_entries
    avg_train_acc = sum(sum_correct.values()) / (total_entries * args.num_labels)
    print(f'\nTrain loss: {avg_train_loss:.4f} and accuracy: {avg_train_acc:.4f}')

    # Write log file
    with open(output_train_log_file_path, 'a+') as fw:
        fw.write(f'{str(datetime.datetime.now())} Train loss: {avg_train_loss:.4f} and ' +
                 f'accuracy: {avg_train_acc:.4f}\n')
    
    return avg_train_loss, avg_train_acc

#%% Validation function
def val_epoch_f(args, model, criterion, dev_dl, device):
    model.eval()
    sum_correct = {x:0 for x in range (0,args.num_labels)}
    sum_val_loss = 0
    total_entries = 0

    for X_bert_encoding, X_transf_mask, Y_labels in tqdm(dev_dl, desc = 'Validation'):
        
        # Move to cuda
        if next(model.parameters()).is_cuda:
            X_bert_encoding = X_bert_encoding.to(device)
            X_transf_mask = X_transf_mask.to(device)
            Y_labels = Y_labels.to(device)
                    
        # Compute predictions:
        with torch.no_grad():
            pred = model(X_bert_encoding, X_transf_mask)
            loss = criterion(pred, Y_labels)
        
        # Book-keeping
        current_batch_size = X_bert_encoding.size()[0]
        total_entries += current_batch_size
        sum_val_loss += (loss.item() * current_batch_size)
        pred = torch.round(pred)
        for idx in range (0, args.num_labels):
            pred_1_label = pred[:, idx]
            Y_1_label = Y_labels[:, idx]
            sum_correct[idx] = (pred_1_label == Y_1_label).sum().item()
        
    avg_val_loss = sum_val_loss / total_entries
    avg_val_accuracy = sum(sum_correct.values()) / (total_entries * args.num_labels)
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
    parser.add_argument('--max_n_pars', default = None, type = int, required = True,
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
   
    # Instantiate dataclasses
    train_dataset = ECHR2_dataset(model_train)
    dev_dataset = ECHR2_dataset(model_dev)

    # Instantiate dataloaders
    train_dl = DataLoader(train_dataset, batch_size = args.batch_size,
                          shuffle = eval(args.shuffle_train), drop_last = False)
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

        if eval(args.save_model_steps) == True and epoch % 10 == 0:
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
    model_params = {'input_dir': args.input_dir,
                    'n_epochs': args.n_epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.lr,
                    'wd': args.wd,
                    'dropout': args.dropout,
                    'momentum': args.momentum,
                    'seed': args.seed,
                    'seq_len': args.seq_len,
                    'num_labels': args.num_labels,              
                    'hidden_dim': args.hidden_dim,
                    'max_n_pars': args.max_n_pars,
                    'pad_idx': args.pad_idx,
                    'save_final_model': args.save_final_model,
                    'save_model_steps': args.save_model_steps,
                    'use_cuda': args.use_cuda,
                    'gpu_ids': args.gpu_ids}
    
    with open(output_path_params, 'w') as fw:
        json.dump(model_params, fw)
    
if __name__ == "__main__":
    main()
