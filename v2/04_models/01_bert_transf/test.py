# v0 -> base
# v1 -> imports module based on config script
# v2 -> includes weighs in output

#%% Imports

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from model_v4 import ECHR2_dataset, ECHR2_model

# Test function
def test_f(args):
    # Load holdout data
    model_test = pd.read_pickle(args.path_model_holdout)
    
    # Instantiate datasets
    test_dataset = ECHR2_dataset(model_test)
    
    # Instantiate dataloader
    test_dl = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)
    
    # Instantiate model and load model weights
    device = torch.device(args.gpu_id)
    model = ECHR2_model(args)
    
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

    for X_facts_ids, X_facts_token_types, X_facts_attn_masks, Y_labels in \
        tqdm(test_dl, desc = 'Testing'):

        # Move to cuda
        X_facts_ids = X_facts_ids.to(device)
        #X_facts_token_types = X_facts_token_types.to(device)
        X_facts_attn_masks = X_facts_attn_masks.to(device)
        Y_labels = Y_labels.to(device)
        
        # Compute predictions and append
        with torch.no_grad():
            pred_batch_score = model(X_facts_ids, X_facts_token_types, X_facts_attn_masks)

        pred_batch_binary = torch.round(pred_batch_score)
        Y_predicted_score.append(pred_batch_score)
        Y_predicted_binary.append(pred_batch_binary)
        Y_ground_truth.append(Y_labels)

    Y_predicted_score = torch.cat(Y_predicted_score, dim = 0).cpu()
    Y_predicted_binary = torch.cat(Y_predicted_binary, dim = 0).cpu()
    Y_ground_truth = torch.cat(Y_ground_truth, dim = 0).cpu()
            
    return Y_predicted_score, Y_predicted_binary, Y_ground_truth
            
def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default = None, type = str, required = True,
                       help = 'input data folder')
    parser.add_argument('--work_dir', default = None, type = str, required = True,
                       help = 'work folder')
    parser.add_argument('--batch_size', default = None, type = int, required = True,
                       help = 'train batch size')
    parser.add_argument('--seq_len', default = None, type = int, required = True,
                       help = 'text sequence length')
    parser.add_argument('--num_labels', default = None, type = int, required = True,
                       help = 'number of labels')
    parser.add_argument('--n_heads', default = None, type = int, required = True,
                       help = 'number of heads in transformer model')
    parser.add_argument('--hidden_dim', default = None, type = int, required = True,
                       help = 'lstm hidden dimension')
    parser.add_argument('--max_n_pars', default = None, type = int, required = True,
                       help = 'maximum number of paragraphs considered')
    parser.add_argument('--pad_idx', default = None, type = int, required = True,
                       help = 'pad token index')  
    parser.add_argument('--gpu_id', default = None, type = int, required = True,
                       help = 'gpu id for testing')
    args = parser.parse_args()
    args.dropout = 0.4
    
    # Path initialization 
#    args.path_model_holdout = os.path.join(args.input_dir, 'model_test.pkl')
    args.path_model_holdout = os.path.join(args.input_dir, 'model_test.pkl')
    args.path_model = os.path.join(args.work_dir, 'model.pt')
    args.path_results_train = os.path.join(args.work_dir, 'train_results.json')
    args.path_results_full = os.path.join(args.work_dir, 'full_results_test.json')
    
    # Compute predictions
    Y_predicted_score, Y_predicted_binary, Y_ground_truth = test_f(args)
    
    # Compute and print metrics
    print(classification_report(Y_ground_truth, Y_predicted_binary))
       
    # Apend metrics to results json file
    with open(args.path_results_train, 'r') as fr:
        results = json.load(fr)
    
    results['Y_test_ground_truth'] = Y_ground_truth.numpy().tolist()
    results['Y_test_prediction_scores'] = Y_predicted_score.numpy().tolist()
    results['Y_test_prediction_binary'] = Y_predicted_binary.numpy().tolist()
    
    # Save metrics
    with open(args.path_results_full, 'w') as fw:
        results = json.dump(results, fw)
    
if __name__ == "__main__":
    main()         
