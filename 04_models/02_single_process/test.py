#%% Imports

import os
import json
import pickle
import datetime
import argparse
import pandas as pd
from tqdm import tqdm
import torch
####
import torch.nn as nn
####
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from attention.model_attn_v16 import ECHR_dataset, ECHR_model

# Test function
def test_f(args):
    # Load holdout data
    model_test = pd.read_pickle(args.path_model_holdout)
    
    # Load embeddings
    with open(args.path_embed, 'rb') as fr:
        id_2_embed = pickle.load(fr)
    
    # Instantiate datasets
    test_dataset = ECHR_dataset(model_test)
    
    # Instantiate dataloader
    test_dl = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)
    
    # Instantiate model and load model weights
    pretrained_embeddings = torch.FloatTensor(list(id_2_embed.values()))
    device = torch.device(args.gpu_id)
    model = ECHR_model(args, pretrained_embeddings)
    
    ####
    #model = nn.DataParallel(model, device_ids = [args.gpu_id])
    model.load_state_dict(torch.load(args.path_model))
    #model.module.load_state_dict(torch.load(args.path_model))
    ####
    
    model.to(device)
    model.eval()
    
    # Test procedure
    Y_predicted_score = []
    Y_predicted_binary = []
    Y_ground_truth = []

    for X_art, X_case, Y in tqdm(test_dl, desc = 'Testing'):
        # Move to cuda
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
            
def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default = None, type = str, required = True,
                       help = 'input data folder')
    parser.add_argument('--work_dir', default = None, type = str, required = True,
                       help = 'work folder')
    parser.add_argument('--path_embed', default = None, type = str, required = True,
                       help = 'path to file with embeddings')   
    parser.add_argument('--batch_size', default = None, type = int, required = True,
                       help = 'train batch size')
    parser.add_argument('--seq_len', default = None, type = int, required = True,
                       help = 'text sequence length')
    parser.add_argument('--num_passages', default = None, type = int, required = True,
                       help = 'number of passages considered')
    parser.add_argument('--embed_dim', default = None, type = int, required = True,
                       help = 'embedding dimension')
    parser.add_argument('--hidden_dim', default = None, type = int, required = True,
                       help = 'lstm hidden dimension')
    parser.add_argument('--att_dim', default = None, type = int, required = True,
                       help = 'attention layer dimension')
    parser.add_argument('--pad_idx', default = None, type = int, required = True,
                       help = 'pad token index')  
    parser.add_argument('--gpu_id', default = None, type = int, required = True,
                       help = 'gpu id for testing')
    args = parser.parse_args()
    args.dropout = 0.4
    
    # Path initialization
    args.path_model_holdout = os.path.join(args.input_dir, 'model_test.pkl')
    args.path_model = os.path.join(args.work_dir, 'model.pt')
    args.path_results_train = os.path.join(args.work_dir, 'train_results.json')
    args.path_results_full = os.path.join(args.work_dir, 'full_results.json')
    
    # Compute predictions
    Y_predicted_score, Y_predicted_binary, Y_ground_truth = test_f(args)
    
    # Compute and print metrics
    tn, fp, fn, tp = confusion_matrix(Y_ground_truth, Y_predicted_binary).ravel()
    precision_model = tp / (tp + fp)
    recall_model = tp / (tp + fn)
    f1_model = 2 * (precision_model * recall_model) / (precision_model + recall_model)
    auc_model = roc_auc_score(Y_ground_truth, Y_predicted_score)

    print(f'\nPrecision model: {precision_model:.4f}')
    print(f'Recall model: {recall_model:.4f}')
    print(f'F1 model: {f1_model:.4f}')
    print(f'AUC model: {auc_model:.4f}\n')
        
    # Apend results to results json file
    with open(args.path_results_train, 'r') as fr:
        results = json.load(fr)
    
    results['Y_test_ground_truth'] = Y_ground_truth
    results['Y_test_prediction_scores'] = Y_predicted_score
    results['Y_test_prediction_binary'] = Y_predicted_binary
    
    with open(args.path_results_full, 'w') as fw:
        results = json.dump(results, fw)   
    
if __name__ == "__main__":
    main()         
