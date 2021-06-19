# Imports
import os
import json
import boto3
import pickle
import datetime
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from model_full import LM_dataset, LM_model

# Test function
def test_f(args):
    # Load holdout data
    model_test = pd.read_pickle(args.path_model_holdout)
    
    # Policy embeddings
    s3 = boto3.resource('s3') 
    bytes = s3.Object(args.aws_bucket_name, args.path_emb_leaf_nodes).get()['Body'].read() 
    fast_text_leaf_nodes_dict = pickle.loads(bytes)
    
    # Loss description embeddings
    print(f'{datetime.datetime.now()} Loading embeddings')
    s3 = boto3.resource('s3') 
    bytes = s3.Object(args.aws_bucket_name, args.path_emb_loss_desc).get()['Body'].read() 
    fast_text_loss_desc_dict = pickle.loads(bytes)
    
    # Dictionary embedding extraction
    id_2_embed_leaf_nodes = fast_text_leaf_nodes_dict['id_to_embedding_mapping']
    id_2_embed_loss_desc = fast_text_loss_desc_dict['id_to_embedding_mapping']  
    print(f'{datetime.datetime.now()} Done')
    
    # Instantiate datasets
    test_dataset = LM_dataset(model_test)
    
    # Instantiate dataloader
    test_dl = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)
    
    # Instantiate model and load model weights
    str_vars = model_test.columns.drop(['LOSS_DESCRIPTION', 'LEAF_NODE_TEXTS', 'full_denial'])
    num_str_vars = len(str_vars)
    pretrained_embeddings_leaf_nodes = torch.FloatTensor(list(id_2_embed_leaf_nodes.values()))
    pretrained_embeddings_loss_desc = torch.FloatTensor(list(id_2_embed_loss_desc.values()))
    device = torch.device(args.gpu_id)
    model = LM_model(args,
                    pretrained_embeddings_leaf_nodes,
                    pretrained_embeddings_loss_desc, num_str_vars)
    model.load_state_dict(torch.load(args.path_model))
    model.to(device)
    model.eval()
    
    # Test procedure
    Y_predicted_score = []
    Y_predicted_binary = []
    Y_ground_truth = []

    for X_str, X_loss_desc, X_leaf_node_texts, Y in tqdm(test_dl, desc = 'Testing'):
        # Move to cuda
        X_str = X_str.cuda(device)
        X_loss_desc = X_loss_desc.cuda(device)
        X_leaf_node_texts = X_leaf_node_texts.cuda(device)
        Y = Y.cuda(device)
        
        # Compute predictions and append
        with torch.no_grad():
            pred_batch_score = model(X_str, X_loss_desc, X_leaf_node_texts).view(-1)
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
    parser.add_argument('--aws_bucket_name', default = None, type = str, required = True,
                       help = 'aws bucket name')
    parser.add_argument('--path_emb_leaf_nodes', default = None, type = str, required = True,
                       help = 'path to file with leaf node embeddings')
    parser.add_argument('--path_emb_loss_desc', default = None, type = str, required = True,
                       help = 'path to file with loss description embeddings')   
    parser.add_argument('--batch_size', default = None, type = int, required = True,
                       help = 'train batch size')
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
    parser.add_argument('--gpu_id', default = None, type = int, required = True,
                       help = 'gpu id for testing')
    args = parser.parse_args()
    args.dropout = 0.4
    
    # Path initialization
    args.path_model_holdout = os.path.join(args.input_dir, 'holdout_model.pkl')
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
