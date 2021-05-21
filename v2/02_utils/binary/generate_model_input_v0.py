#%% Imports
import os
import pandas as pd
from tqdm import tqdm

#%% Path definition
input_folder ='C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/v2/01_preprocessed/01_50pars_256_tok/02_full_binary'
output_folder = input_folder

path_input_train = os.path.join(input_folder, 'tokenized_train.pkl')
path_input_dev = os.path.join(input_folder, 'tokenized_dev.pkl')
path_input_test = os.path.join(input_folder, 'tokenized_test.pkl')
path_input_echr = os.path.join(input_folder, 'tokenized_echr.pkl')

path_output_train = os.path.join(output_folder, 'model_train.pkl')
path_output_dev = os.path.join(output_folder, 'model_dev.pkl')
path_output_test = os.path.join(output_folder, 'model_test.pkl')

#%% Function definitions
# Dataset preprocessing
def preprocess_dataset_f(dataset, echr_set, labels_to_skip, id_2_label):
    # Facts initialization
    facts_identifiers = []
    facts_ids = []
    facts_token_type = []
    facts_att_mask = []
    # ECHR articles initialization
    echr_identifiers = []
    echr_ids = []
    echr_token_type = []
    echr_att_mask = []
    # Label initialization
    label_binary = []
    
    for idx, row in tqdm(dataset.iterrows(), total = len(dataset)):
        for label in id_2_label.values():
            if label in labels_to_skip:
                continue
            else:
                # Append facts
                facts_identifiers.append(idx)
                facts_ids.append(row['input_ids'])
                facts_token_type.append(row['token_type'])
                facts_att_mask.append(row['att_mask'])
                # Append ECHR articles
                selected_echr_art = echr_set[echr_set['art_ids'] == label]
                assert(len(selected_echr_art) == 1)
                echr_identifiers.append(label)
                echr_ids.append(selected_echr_art['input_ids'].item())
                echr_token_type.append(selected_echr_art['token_type'].item())
                echr_att_mask.append(selected_echr_art['att_mask'].item())
                if label in row['labels']:
                    label_binary.append(1)
                else:
                    label_binary.append(0)
    
    output_df = pd.DataFrame({'facts_identifiers': facts_identifiers,
                              'facts_ids': facts_ids,
                              'facts_token_type': facts_token_type,
                              'facts_att_mask': facts_att_mask,
                              'echr_identifiers': echr_identifiers,
                              'echr_ids': echr_ids,
                              'echr_token_type': echr_token_type,
                              'echr_att_mask': echr_att_mask,
                              'label': label_binary
                              })
    return output_df

#%% Global initialization
toy_data = False
len_toy_data = 100
labels_to_skip = ['P3-1']
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

#%% Data load
train_set = pd.read_pickle(path_input_train)
dev_set = pd.read_pickle(path_input_dev)
test_set = pd.read_pickle(path_input_test)
echr_set = pd.read_pickle(path_input_echr)

if toy_data == True:
    train_set = train_set[0:len_toy_data]
    dev_set = dev_set[0:len_toy_data]
    test_set = test_set[0:len_toy_data]

#%% Preprocess datasets
model_train_df = preprocess_dataset_f(train_set, echr_set, labels_to_skip, id_2_label)
model_dev_df = preprocess_dataset_f(dev_set, echr_set, labels_to_skip, id_2_label)
model_test_df = preprocess_dataset_f(test_set, echr_set, labels_to_skip, id_2_label)

#%% Save output files
model_train_df.to_pickle(path_output_train)
model_dev_df.to_pickle(path_output_dev)
model_test_df.to_pickle(path_output_test)
