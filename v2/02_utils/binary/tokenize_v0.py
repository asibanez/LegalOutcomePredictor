# v0 -> BERT - tokenizes facts and ECHR articles

#%% Imports
import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

#%% Function definitions

# Paragraph tokenization
def tokenize_f(paragraphs, seq_len, max_n_pars):
    # List initialization
    input_id_list = []
    token_type_id_list = []
    attention_mask_list = []
    # Tokenization of empty fact
    empty_fact_tokens = bert_tokenizer('',
                                       return_tensors = 'pt',
                                       padding = 'max_length',
                                       truncation = True,
                                       max_length = seq_len)

    empty_fact_input_id = empty_fact_tokens['input_ids'].squeeze(0).type(torch.LongTensor)
    empty_fact_token_type_id = empty_fact_tokens['token_type_ids'].squeeze(0).type(torch.LongTensor)
    empty_fact_attn_mask = empty_fact_tokens['attention_mask'].squeeze(0).type(torch.LongTensor)
    
    for idx, paragraph_set in enumerate(tqdm(paragraphs, desc = 'Tokenizing dataset')):
        aux_input_id_list = []
        aux_token_type_id_list = []
        aux_attention_mask_list = []
        for paragraph in paragraph_set[0:max_n_pars]:
            fact_tokens = bert_tokenizer(paragraph,
                                         return_tensors = 'pt',
                                         padding = 'max_length',
                                         truncation = True,
                                         max_length = seq_len)
            aux_input_id_list.append(fact_tokens['input_ids'].squeeze(0).type(torch.LongTensor))
            aux_token_type_id_list.append(fact_tokens['token_type_ids'].squeeze(0).type(torch.LongTensor))
            aux_attention_mask_list.append(fact_tokens['attention_mask'].squeeze(0).type(torch.LongTensor))
        
        #Padding
        n_pad_items = max_n_pars - len(paragraph_set)
        
        aux_input_id_list += [empty_fact_input_id for x in range(n_pad_items)]
        aux_token_type_id_list += [empty_fact_token_type_id for x in range(n_pad_items)]
        aux_attention_mask_list += [empty_fact_attn_mask for x in range(n_pad_items)]
        
        # Convert list of tensors to tensor
        aux_input_id_tensor = torch.cat(aux_input_id_list)
        aux_token_type_id_tensor = torch.cat(aux_token_type_id_list)
        aux_attention_mask_tensor = torch.cat(aux_attention_mask_list)
    
        # Append to main list
        input_id_list.append(aux_input_id_tensor)
        token_type_id_list.append(aux_token_type_id_tensor)
        attention_mask_list.append(aux_attention_mask_tensor)
        
    return input_id_list, token_type_id_list, attention_mask_list

#%% Path definition
echr_input_path = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/v2/00_ECHR_law/01_parsed/ECHR_paragraphs_final_no_bullets.csv'
output_folder = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/v2/01_preprocessed/01_50pars_256_tok/02_full_binary'
#echr_input_path = '/data####'
#output_folder = '/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/v2/01_preprocessed/01_50pars_256_tok/01_full_binary'
train_set_path = os.path.join(output_folder, 'model_train.pkl')
val_set_path = os.path.join(output_folder, 'model_dev.pkl')
test_set_path = os.path.join(output_folder, 'model_test.pkl')
echr_set_path = os.path.join(output_folder, 'echr.pkl')

#%% Global initialization
num_labels = 33
seq_len = 256
max_n_pars_facts = 50
max_n_pars_echr = 6
pad_int = 0
toy_data = False
len_toy_data = 100

#%% Facts data loading
train_set = load_dataset('ecthr_cases', split = 'train')
val_set = load_dataset('ecthr_cases', split = 'validation')
test_set = load_dataset('ecthr_cases', split = 'test')

if toy_data == True:
    train_set = train_set[0:len_toy_data]
    val_set = val_set[0:len_toy_data]
    test_set = test_set[0:len_toy_data]

train_facts = train_set['facts']
val_facts = val_set['facts']
test_facts = test_set['facts']

train_labels = train_set['labels']
val_labels = val_set['labels']
test_labels = test_set['labels']

#%% ECHR data loading
echr_set = pd.read_csv(echr_input_path)
echr_set = pd.DataFrame(echr_set.groupby(['Art ID'])['Text'].apply(list)).reset_index()
echr_art_ids = echr_set['Art ID']
echr_pars = echr_set['Text']

#%% Tokenizer instantiation
model_name = 'nlpaueb/legal-bert-small-uncased'
bert_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
    
#%% Tokenize datasets
train_input_ids, train_token_type_ids, train_attn_masks = \
    tokenize_f(train_facts, seq_len, max_n_pars_facts)
val_input_ids, val_token_type_ids, val_attn_masks = \
    tokenize_f(val_facts, seq_len, max_n_pars_facts)
test_input_ids, test_token_type_ids, test_attn_masks = \
    tokenize_f(test_facts, seq_len, max_n_pars_facts)

#%% Tokenize ECHR articles
echr_input_ids, echr_token_type_ids, echr_attn_masks = \
    tokenize_f(echr_pars, seq_len, max_n_pars_echr)

#%% Build output dataframes
train_set_df = pd.DataFrame({'input_ids': train_input_ids,
                             'token_type': train_token_type_ids,
                             'att_mask': train_attn_masks,
                             'labels': train_labels})
    
val_set_df = pd.DataFrame({'input_ids': val_input_ids,
                           'token_type': val_token_type_ids,
                           'att_mask': val_attn_masks,
                           'labels': val_labels})

test_set_df = pd.DataFrame({'ids': test_input_ids,
                            'token_type': test_token_type_ids,
                            'att_mask': test_attn_masks,
                            'labels': test_labels})

echr_set_df = pd.DataFrame({'art_ids': echr_art_ids,
                            'input_ids': echr_input_ids,
                            'token_type': echr_token_type_ids,
                            'att_mask': echr_attn_masks})

#%% Save outputs
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
    print("Created folder : ", output_folder)

print(f'Saving datasets to {output_folder}')
train_set_df.to_pickle(train_set_path)
val_set_df.to_pickle(val_set_path)
test_set_df.to_pickle(test_set_path)
echr_set_df.to_pickle(echr_set_path)
print('Done')
