#%% Imports
import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

#%% Path definition
output_folder = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/v2/01_preprocessed/00_toy'
train_set_path = os.path.join(output_folder, 'model_train.pkl')
val_set_path = os.path.join(output_folder, 'model_dev.pkl')
test_set_path = os.path.join(output_folder, 'model_test.pkl')

#%% Global initialization
num_labels = 33
seq_len = 512
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
              16:'19',
              17:'34',
              18:'38',
              19:'39',
              20:'46',              
              21:'P1-1',
              22:'P1-2',
              23:'P1-3',
              24:'P3-1',
              25:'P4-2',
              26:'P4-4',
              27:'P6-3',
              28:'P7-1',
              29:'P7-2',
              30:'P7-3',
              31:'P7-4',
              32:'P7-5',
              33:'P12-1'}

#%% Label dict
label_2_id = {id_2_label[x]:x for x in id_2_label.keys()}

#%% Data load
train_set = load_dataset('ecthr_cases', split = 'train')[0:100]
val_set = load_dataset('ecthr_cases', split = 'validation')[0:100]
test_set = load_dataset('ecthr_cases', split = 'test')[0:100]

train_facts = train_set['facts']
val_facts = val_set['facts']
test_facts = test_set['facts']

train_labels = train_set['labels']
val_labels = val_set['labels']
test_labels = test_set['labels']

#%% Tokenizer instantiation
model_name = 'nlpaueb/legal-bert-small-uncased'
bert_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)

#%% Tokenization
train_facts_tokens = []
for fact_set in tqdm(train_facts, desc = 'tokenizing train set'):
    aux_list = []
    for fact in fact_set:
        fact_tokens = bert_tokenizer(fact,
                                     return_tensors = 'pt',
                                     padding = 'max_length',
                                     truncation = True,
                                     max_length = seq_len)
        aux_list.append(fact_tokens)
    train_facts_tokens.append(aux_list)
    
#%%
val_facts_tokens = []
for fact_set in tqdm(val_facts, desc = 'tokenizing validation set'):
    aux_list = []
    for fact in fact_set:
        fact_tokens = bert_tokenizer(fact,
                                     return_tensors = 'pt',
                                     padding = 'max_length',
                                     truncation = True,
                                     max_length = seq_len)
        aux_list.append(fact_tokens)
    val_facts_tokens.append(aux_list)

#%%
test_facts_tokens = []
for fact_set in tqdm(test_facts, desc = 'tokenizing test set'):
    aux_list = []
    for fact in fact_set:
        fact_tokens = bert_tokenizer(fact,
                                     return_tensors = 'pt',
                                     padding = 'max_length',
                                     truncation = True,
                                     max_length = seq_len)
        aux_list.append(fact_tokens)
    test_facts_tokens.append(aux_list)

#%% Label preprocessing
train_labels_processed = []
for labels in tqdm(train_labels, desc = 'processing train labels'):
    aux_labels = [0] * num_labels
    for label in labels:
        pos = label_2_id[label]
        aux_labels[pos] = 1
    train_labels_processed.append(aux_labels)

#%%
val_labels_processed = []
for labels in tqdm(val_labels, desc = 'processing validation labels'):
    aux_labels = [0] * num_labels
    for label in labels:
        pos = label_2_id[label]
        aux_labels[pos] = 1
    val_labels_processed.append(aux_labels)

#%%
test_labels_processed = []
for labels in tqdm(test_labels, desc = 'processing test labels'):
    aux_labels = [0] * num_labels
    for label in labels:
        pos = label_2_id[label]
        aux_labels[pos] = 1
    test_labels_processed.append(aux_labels)

#%% Build output dataframes
train_set_df = pd.DataFrame({'facts': train_facts_tokens,
                             'labels': train_labels_processed})
    
val_set_df = pd.DataFrame({'facts': val_facts_tokens,
                           'labels': val_labels_processed})

test_set_df = pd.DataFrame({'facts': test_facts_tokens,
                            'labels': test_labels_processed})

#%% Save outputs
train_set_df.to_pickle(train_set_path)
val_set_df.to_pickle(val_set_path)
test_set_df.to_pickle(test_set_path)
