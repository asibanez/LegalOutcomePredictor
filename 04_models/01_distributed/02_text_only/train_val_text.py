# Imports
import os
import json
import boto3
import random
import pickle
import argparse
import datetime
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pudb.remote import set_trace
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from model_text import LM_dataset, LM_model

# Train function
def train_epoch_f(gpu, args):
    # Initialization
    rank = args.nr * args.gpus + gpu	                          
    dist.init_process_group(                                   
    	backend = 'nccl',                                         
   		init_method = 'env://',                                   
    	world_size = args.world_size,                              
    	rank = rank)                     
    torch.manual_seed(args.seed)
    
    # Load train / dev data
    print(f'{datetime.datetime.now()} Loading train / dev data - GPU: {gpu}')
    model_train = pd.read_pickle(args.path_model_train)
    model_dev = pd.read_pickle(args.path_model_dev)
    print(f'{datetime.datetime.now()} Done - GPU: {gpu}')
    
    # Policy embeddings
    print(f'{datetime.datetime.now()} Loading policy embeddings - GPU: {gpu}')
    s3 = boto3.resource('s3') 
    bytes = s3.Object(args.aws_bucket_name, args.path_emb_leaf_nodes).get()['Body'].read() 
    fast_text_leaf_nodes_dict = pickle.loads(bytes)
    print(f'{datetime.datetime.now()} Done - GPU: {gpu}')
    
    # Loss description embeddings
    print(f'{datetime.datetime.now()} Loading loss description embeddings - GPU: {gpu}')
    s3 = boto3.resource('s3') 
    bytes = s3.Object(args.aws_bucket_name, args.path_emb_loss_desc).get()['Body'].read() 
    fast_text_loss_desc_dict = pickle.loads(bytes)
    print(datetime.datetime.now(), 'Done')
    
    # Dictionary embedding extraction
    id_2_embed_leaf_nodes = fast_text_leaf_nodes_dict['id_to_embedding_mapping']
    id_2_embed_loss_desc = fast_text_loss_desc_dict['id_to_embedding_mapping']  
    
    # Instantiate datasets
    train_dataset = LM_dataset(model_train)
    dev_dataset = LM_dataset(model_dev)
    
    # Define data sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas = args.world_size,
    	rank = rank)
        
    # Instantiate dataloader
    train_dl = DataLoader(dataset = train_dataset,
                          batch_size = args.batch_size,
                          shuffle = False,
                          num_workers = 0,
                          pin_memory = True,
                          sampler = train_sampler)
    
    # Instantiate model
    pretrained_embeddings_leaf_nodes = torch.FloatTensor(list(id_2_embed_leaf_nodes.values()))
    pretrained_embeddings_loss_desc = torch.FloatTensor(list(id_2_embed_loss_desc.values()))
    model = LM_model(args,
                     pretrained_embeddings_leaf_nodes,
                     pretrained_embeddings_loss_desc)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    
    # Instantiate optimizer & criterion
    criterion = nn.BCELoss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr,
                                 weight_decay = args.wd)
    
    # Wrap model for parallelization
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])
    
    # Training procedure
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    for idx_ep, epoch in enumerate(tqdm(range(args.n_epochs))):        
        sum_correct = 0
        total_entries = 0
        sum_train_loss = 0
        model.train()
        
        for idx, (X_loss_desc, X_leaf_node_texts, Y) in tqdm(enumerate(train_dl),
                                                             total = len(train_dl),
                                                             desc = 'Training'):    
            # Write log file
            if gpu == 0:
                with open(args.output_train_log_file_path, 'a+') as fw:
                    fw.write(f'Epoch {idx_ep+1:,} of {args.n_epochs:,}\t')
                    fw.write(f'Step {idx+1:,} of {len(train_dl):,}\n')
        
            # Move to cuda
            if next(model.parameters()).is_cuda:
                X_loss_desc = X_loss_desc.cuda()
                X_leaf_node_texts = X_leaf_node_texts.cuda()
                Y = Y.cuda()

            # Zero gradients
            optimizer.zero_grad()

            #Forward + backward + optimize
            pred = model(X_loss_desc, X_leaf_node_texts).view(-1)
            loss = criterion(pred, Y)

            # Backpropagate + model update
            loss.backward()
            optimizer.step()           

            # Book-keeping
            current_batch_size = X_loss_desc.size()[0]
            total_entries += current_batch_size
            sum_train_loss += (loss.item() * current_batch_size)
            pred = torch.round(pred.view(pred.shape[0]))
            sum_correct += (pred == Y).sum().item()   

        avg_train_loss = sum_train_loss / total_entries
        avg_train_acc = sum_correct / total_entries
        if gpu == 0:
            print(f'\ntrain loss: {avg_train_loss:.4f} and accuracy: {avg_train_acc:.4f}')
        
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(avg_train_acc)

        # Save model step and validate
        if gpu == 0:
            if eval(args.save_model_steps) == True:
                torch.save(model.module.state_dict(),
                           args.output_path_model + '.' + str(idx_ep))
            val_loss_history, val_acc_history = validate_f(
                model, args, criterion, dev_dataset, gpu, val_loss_history,val_acc_history)
            
    # Save model and results
    if gpu == 0:
        # Save model
        if eval(args.save_final_model) == True:
            torch.save(model.module.state_dict(), args.output_path_model)
            
        # Save results
        train_results = {'training_loss': train_loss_history,
                         'training_acc': train_acc_history,
                         'validation_loss': val_loss_history,
                         'validation_acc': val_acc_history}
        with open(args.output_path_results, 'w') as fw:
            json.dump(train_results, fw)
    
        # Save model parameters
        model_params = {'n_epochs': args.n_epochs,
                        'batch_size': args.batch_size,
                        'learning_rate': args.lr,
                        'wd': args.wd,
                        'dropout': args.dropout,
                        'momentum': args.momentum,
                        'seed': args.seed,
                        'seq_len': args.seq_len,
                        'num_leaf_nodes': args.num_leaf_nodes,
                        'embed_dim': args.embed_dim,
                        'hidden_dim': args.hidden_dim,
                        'att_dim': args.att_dim,
                        'pad_idx': args.pad_idx,
                        'save_final_model': args.save_final_model,
                        'save_model_steps': args.save_model_steps,
                        'nodes': args.nodes,
                        'gpus': args.gpus,
                        'nr': args.nr}
        with open(args.output_path_params, 'w') as fw:
            json.dump(model_params, fw)

def validate_f(model, args, criterion, dev_dataset, device, val_loss_history, val_acc_history):
    # Initialization
    model.eval()
    # Instantiate dataloader
    dev_dl = DataLoader(dataset = dev_dataset,
                        batch_size = args.batch_size * 2,
                        shuffle = False)
    # Validation
    sum_correct = 0
    total_entries = 0
    sum_val_loss = 0

    for X_loss_desc, X_leaf_node_texts, Y in tqdm(
        dev_dl, total = len(dev_dl), desc = 'Validation'):    
        # Move to cuda
        if next(model.parameters()).is_cuda:
            X_loss_desc = X_loss_desc.cuda(device)
            X_leaf_node_texts = X_leaf_node_texts.cuda(device)
            Y = Y.cuda(device)

        # Compute predictions:
        with torch.no_grad():
            pred = model(X_loss_desc, X_leaf_node_texts).view(-1)
            loss = criterion(pred, Y)
            
        # Book-keeping
        current_batch_size = X_loss_desc.size()[0]
        total_entries += current_batch_size
        sum_val_loss += (loss.item() * current_batch_size)
        pred = torch.round(pred.view(pred.shape[0]))
        sum_correct += (pred == Y).sum().item()   

    avg_val_loss = sum_val_loss / total_entries
    avg_val_acc = sum_correct / total_entries
    val_loss_history.append(avg_val_loss)
    val_acc_history.append(avg_val_acc)
    print(f'validation loss: {avg_val_loss:.4f} and accuracy: {avg_val_acc:.4f}')
    
    return val_loss_history, val_acc_history
            
def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default = None, type = str, required = True,
                       help = 'input folder')
    parser.add_argument('--output_dir', default = None, type = str, required = True,
                       help = 'output folder')
    parser.add_argument('--aws_bucket_name', default = None, type = str, required = True,
                       help = 'aws bucket name')
    parser.add_argument('--path_emb_leaf_nodes', default = None, type = str, required = True,
                       help = 'path to file with leaf node embeddings')
    parser.add_argument('--path_emb_loss_desc', default = None, type = str, required = True,
                       help = 'path to file with loss description embeddings')   
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
    parser.add_argument('--num_leaf_nodes', default = None, type = int, required = True,
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
    parser.add_argument('--nodes', default = 1, type = int, metavar = 'N',
                        required = True, help = 'number of nodes used for training')
    parser.add_argument('--gpus', default = 1, type = int, required = True,
                        help='number of gpus per node')
    parser.add_argument('--nr', default = 0, type = int, required = True,
                        help = 'ranking within the nodes')
    args = parser.parse_args()
              
    # Path initialization
    args.path_model_train = os.path.join(args.input_dir, 'train_model.pkl')
    args.path_model_dev = os.path.join(args.input_dir, 'dev_model.pkl')
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        print("Created folder : ", args.output_dir)

    args.output_train_log_file_path = os.path.join(args.output_dir, 'train_log.txt')
    args.output_path_model = os.path.join(args.output_dir, 'model.pt')
    args.output_path_results = os.path.join(args.output_dir, 'train_results.json')
    args.output_path_params = os.path.join(args.output_dir, 'params.json')     
    args.world_size = args.gpus * args.nodes

    # Multi GPU training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    mp.spawn(train_epoch_f, nprocs = args.gpus, args = (args,))
    
if __name__ == "__main__":
    main()         
