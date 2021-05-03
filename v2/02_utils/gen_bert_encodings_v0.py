# v0

#%% Imports
import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModel

#%% Path definition
#input_folder = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/v2/01_preprocessed/00_toy'
#output_folder = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/v2/01_preprocessed/02_toy_bert'
input_folder = '/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/v2/01_preprocessed/02_toy_1'
output_folder = '/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/v2/01_preprocessed/03_toy_1_bert'

input_train_set_path = os.path.join(input_folder, 'model_train.pkl')
input_dev_set_path = os.path.join(input_folder, 'model_dev.pkl')
input_test_set_path = os.path.join(input_folder, 'model_test.pkl')

output_train_set_path = os.path.join(output_folder, 'model_train.pkl')
output_dev_set_path = os.path.join(output_folder, 'model_dev.pkl')
output_test_set_path = os.path.join(output_folder, 'model_test.pkl')

#%% Function definitions
# Encode paragraphs - BERT encodings & masks for transformers
def encode_par_f(dataset, bert_model, device_id):

    bert_encoding_list = []
    transf_mask_list = []
    empty_par_ids = torch.cat([torch.tensor([101,102]),torch.zeros(510)]).long().to(device_id)
    
    for row_idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc='iterating over samples'):
        X_facts_ids = row['facts_ids'].to(device_id)
        X_facts_token_types = row['facts_token_type'].to(device_id)
        X_facts_attn_masks = row['facts_att_mask'].to(device_id)
        
        bert_out = {}
        transf_mask = torch.zeros(max_n_pars, dtype=torch.bool)             # max_n_pars
        
        for idx in tqdm(range(0, max_n_pars), desc = 'Iterating through paragraphs'):   
            span_b = seq_len * idx
            span_e = seq_len * (idx + 1)
            
            # Slice sequence
            facts_ids = X_facts_ids[span_b:span_e]                          # seq_len
            facts_token_types = X_facts_token_types[span_b:span_e]          # seq_len
            facts_attn_masks = X_facts_attn_masks[span_b:span_e]            # seq_len
            
            # Generate masks for transformer
            if torch.equal(facts_ids, empty_par_ids):
                transf_mask[idx] = True                                     # 1
            
            # Generate input dict to bert model
            bert_input = {'input_ids': facts_ids.unsqueeze(0),              # 1 x seq_len
                          'token_type_ids': facts_token_types.unsqueeze(0), # 1 x seq_len
                          'attention_mask': facts_attn_masks.unsqueeze(0)}  # 1 x seq_len
                  
            # Compute 
            output = bert_model(**bert_input, output_hidden_states = True)
            bert_out[idx] = output['pooler_output']                         # 1 x h_dim
    
        # Stack paragraph outputs    
        bert_out = torch.cat(list(bert_out.values()), dim = 0)              # max_n_pars x h_dim
        
        # Append to main lists
        bert_encoding_list.append(bert_out)
        transf_mask_list.append(transf_mask)
    
    # Generate output dataframe
    output_df = pd.Dataframe({'bert_encoding': bert_encoding_list,
                              'transf_mask': transf_mask_list,
                              'labels': dataset.labels})

    return output_df

#%% Global initialization
max_n_pars = 200
seq_len = 512
device_id = 0

#%% Read dataset
print('Loading datasets')
train_dataset = pd.read_pickle(input_train_set_path)
dev_dataset = pd.read_pickle(input_dev_set_path)
test_dataset = pd.read_pickle(input_test_set_path)
print('Done')

#%% Define model
model_name = 'nlpaueb/legal-bert-small-uncased'
bert_model = AutoModel.from_pretrained(model_name)
bert_model.to(device_id)
# Freeze bert parameters
for parameter in bert_model.parameters():
    parameter.requires_grad = False

#%% Process datasets
train_dataset_bert = encode_par_f(train_dataset, bert_model, device_id)
dev_dataset_bert = encode_par_f(dev_dataset, bert_model, device_id)
test_dataset_bert = encode_par_f(test_dataset, bert_model, device_id)

#%% Save datasets
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
    print("Created folder : ", output_folder)

pd.to_pickle(train_dataset, output_train_set_path)
pd.to_pickle(dev_dataset, output_dev_set_path)
pd.to_pickle(test_dataset, output_test_set_path)
                         
