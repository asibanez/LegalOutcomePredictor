# v1 -> Saves info as stacked tensors
# v2 -> Saves info as flat tensors
# v3 -> Adds padding to tensors
# v4 -> Tokenizer outputs converted to long tensors
# v5 -> Empty paragraphs as per BERT tokenizer & tokenization and 
#       label processing moved to functions
#       Creates output folder if not exists

#%% Imports
import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

#%% Function definitions

# Tokenization
def tokenize_f(facts):
    # List initialization
    facts_input_id_list = []
    facts_token_type_id_list = []
    facts_attention_mask_list = []
    # Tokenization of empty fact
    empty_fact_tokens = bert_tokenizer('',
                                       return_tensors = 'pt',
                                       padding = 'max_length',
                                       truncation = True,
                                       max_length = seq_len)

    empty_fact_input_id = empty_fact_tokens['input_ids'].squeeze(0).type(torch.LongTensor)
    empty_fact_token_type_id = empty_fact_tokens['token_type_ids'].squeeze(0).type(torch.LongTensor)
    empty_fact_attn_mask = empty_fact_tokens['attention_mask'].squeeze(0).type(torch.LongTensor)
    
    for idx, fact_set in enumerate(tqdm(facts, desc = 'Tokenizing dataset')):
        aux_input_id_list = []
        aux_token_type_id_list = []
        aux_attention_mask_list = []
        for fact in fact_set[0:max_n_pars]:
            fact_tokens = bert_tokenizer(fact,
                                         return_tensors = 'pt',
                                         padding = 'max_length',
                                         truncation = True,
                                         max_length = seq_len)
            aux_input_id_list.append(fact_tokens['input_ids'].squeeze(0).type(torch.LongTensor))
            aux_token_type_id_list.append(fact_tokens['token_type_ids'].squeeze(0).type(torch.LongTensor))
            aux_attention_mask_list.append(fact_tokens['attention_mask'].squeeze(0).type(torch.LongTensor))
        
        #Padding
        n_pad_items = max_n_pars - len(fact_set)
        
        aux_input_id_list += [empty_fact_input_id for x in range(n_pad_items)]
        aux_token_type_id_list += [empty_fact_token_type_id for x in range(n_pad_items)]
        aux_attention_mask_list += [empty_fact_attn_mask for x in range(n_pad_items)]
        
        # Convert list of tensors to tensor
        aux_input_id_tensor = torch.cat(aux_input_id_list)
        aux_token_type_id_tensor = torch.cat(aux_token_type_id_list)
        aux_attention_mask_tensor = torch.cat(aux_attention_mask_list)
    
        # Append to main list
        facts_input_id_list.append(aux_input_id_tensor)
        facts_token_type_id_list.append(aux_token_type_id_tensor)
        facts_attention_mask_list.append(aux_attention_mask_tensor)
        
    return facts_input_id_list, facts_token_type_id_list, facts_attention_mask_list

#%% Label preprocessing
def label_process_f(labels):
    labels_processed = []
    for labels in tqdm(labels, desc = 'Processing labels'):
        aux_labels = [0] * num_labels
        for label in labels:
            pos = label_2_id[label]
            aux_labels[pos] = 1
        labels_processed.append(aux_labels)
    
    return labels_processed

#%% Path definition
#output_folder = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/v2/01_preprocessed/04_toy_4'
output_folder = '/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/v2/01_preprocessed/01_full_1'
train_set_path = os.path.join(output_folder, 'model_train.pkl')
val_set_path = os.path.join(output_folder, 'model_dev.pkl')
test_set_path = os.path.join(output_folder, 'model_test.pkl')

#%% Global initialization
num_labels = 33
seq_len = 512
max_n_pars = 200
pad_int = 0
padded_len = seq_len * max_n_pars
id_2_label = {0: '2',
              1: '3',
              2: '4',
              3: '5',
              4: '6',
              5: '7',
              6: '8',
              7: '9',
              8: '10',
              9: '11',
              10:'12',
              11:'13',
              12:'14',
              13:'15',
              14:'17',
              15:'18',
              16:'34',
              17:'38',
              18:'39',
              19:'46',              
              20:'P1-1',
              21:'P1-2',
              22:'P1-3',
              23:'P3-1',
              24:'P4-2',
              25:'P4-4',
              26:'P6-3',
              27:'P7-1',
              28:'P7-2',
              29:'P7-3',
              30:'P7-4',
              31:'P7-5',
              32:'P12-1'}

#%% Label dict
label_2_id = {id_2_label[x]:x for x in id_2_label.keys()}

#%% Data load
train_set = load_dataset('ecthr_cases', split = 'train')        #[0:100]
val_set = load_dataset('ecthr_cases', split = 'validation')     #[0:100]
test_set = load_dataset('ecthr_cases', split = 'test')          #[0:100]

train_facts = train_set['facts']
val_facts = val_set['facts']
test_facts = test_set['facts']

train_labels = train_set['labels']
val_labels = val_set['labels']
test_labels = test_set['labels']

#%% Tokenizer instantiation
model_name = 'nlpaueb/legal-bert-small-uncased'
bert_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
    
#%% Tokenize datasets
train_input_ids, train_token_type_ids, train_attn_masks = \
    tokenize_f(train_facts)
val_input_ids, val_token_type_ids, val_attn_masks = \
    tokenize_f(val_facts)
test_input_ids, test_token_type_ids, test_attn_masks = \
    tokenize_f(test_facts)

#%% Label preprocessing
train_labels_processed = label_process_f(train_labels)
val_labels_processed = label_process_f(val_labels)
test_labels_processed = label_process_f(test_labels)

#%% Build output dataframes
train_set_df = pd.DataFrame({'facts_ids': train_input_ids,
                             'facts_token_type': train_token_type_ids,
                             'facts_att_mask': train_attn_masks,
                             'labels': train_labels_processed})
    
val_set_df = pd.DataFrame({'facts_ids': val_input_ids,
                           'facts_token_type': val_token_type_ids,
                           'facts_att_mask': val_attn_masks,
                           'labels': val_labels_processed})

test_set_df = pd.DataFrame({'facts_ids': test_input_ids,
                            'facts_token_type': test_token_type_ids,
                            'facts_att_mask': test_attn_masks,
                            'labels': test_labels_processed})

#%% Save outputs
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
    print("Created folder : ", output_folder)

print(f'Saving datasets to {output_folder}')
train_set_df.to_pickle(train_set_path)
val_set_df.to_pickle(val_set_path)
test_set_df.to_pickle(test_set_path)
print('Done')
