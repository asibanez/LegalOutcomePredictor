# WORKING VERSION

#%% Imports
import os
import tqdm
import pickle
import pandas as pd

#%% Path definition

base_folder = os.getcwd()
input_folder = os.path.join(base_folder, '01_data', '01_preprocessed')
output_path = os.path.join(base_folder, '01_data', '01_preprocessed', 'data_map_df')

train_dev_test_files = ['case_EN_train', 'case_EN_dev', 'case_EN_test']
ECHR_dict_path = os.path.join(input_folder, 'ECHR.dict')
case_train_path = os.path.join(input_folder, 'case_EN_train')
case_dev_path = os.path.join(input_folder, 'case_EN_dev')
case_test_path = os.path.join(input_folder, 'case_EN_test')

#%% Variable initializatoin

arts_to_skip = ['P1']

#%% Load data

with open(ECHR_dict_path, 'rb') as fr:
    ECHR_dict = pickle.load(fr)

with open(case_train_path, 'rb') as fr:
    case_train_df = pickle.load(fr)

with open(case_dev_path, 'rb') as fr:
    case_dev_df = pickle.load(fr)

with open(case_test_path, 'rb') as fr:
    case_test_df = pickle.load(fr)

#%% Merge case dataframes

print('shape case train = ', case_train_df.shape)
print('shape case dev = ', case_dev_df.shape)
print('shape case test = ', case_test_df.shape)

#%% Restructure train data

str_data_train = pd.DataFrame(columns = ['case_id', 'article_id', 'broken'])

for case in tqdm.tqdm(case_test_df.iterrows()):
    case_id, article_id, broken = [], [], []
    
    violated_art_ids = case[1]['VIOLATED_ARTICLES']
    case_id = [case[1]['ITEMID']] * 67
    art_id = list(range(1,68))
    outcome = [0] * 67
    
    for violated_art_id in violated_art_ids:
        ### Skip unwanted articles
        if violated_art_id in arts_to_skip: continue
        violated_art_id = int(violated_art_id)
        outcome[violated_art_id] = 1
        
    aux_df = pd.DataFrame({'case_id':case_id,
                           'article_id': art_id,
                           'broken': outcome})
                           
    str_data_train = pd.concat([str_data_train, aux_df], axis = 0,
                               ignore_index=True)
    
str_data_train['dataset'] = ['train'] * len(str_data_train)
    
#%% Restructure dev data

str_data_dev = pd.DataFrame(columns = ['case_id', 'article_id', 'broken'])

for case in tqdm.tqdm(case_dev_df.iterrows()):
    case_id, article_id, broken = [], [], []
    
    violated_art_ids = case[1]['VIOLATED_ARTICLES']
    case_id = [case[1]['ITEMID']] * 67
    art_id = list(range(1,68))
    outcome = [0] * 67
    
    for violated_art_id in violated_art_ids:
        ### Skip unwanted articles
        if violated_art_id in arts_to_skip: continue
        violated_art_id = int(violated_art_id)
        outcome[violated_art_id] = 1
        
    aux_df = pd.DataFrame({'case_id':case_id,
                           'article_id': art_id,
                           'broken': outcome})
                           
    str_data_dev = pd.concat([str_data_dev, aux_df], axis = 0,
                             ignore_index=True)

str_data_dev['dataset'] = ['dev'] * len(str_data_dev)

#%% Restructure test data

str_data_test = pd.DataFrame(columns = ['case_id', 'article_id', 'broken'])

for case in tqdm.tqdm(case_dev_df.iterrows()):
    case_id, article_id, broken = [], [], []
    
    violated_art_ids = case[1]['VIOLATED_ARTICLES']
    case_id = [case[1]['ITEMID']] * 67
    art_id = list(range(1,68))
    outcome = [0] * 67
    
    for violated_art_id in violated_art_ids:
        ### Skip unwanted articles
        if violated_art_id in arts_to_skip: continue
        violated_art_id = int(violated_art_id)
        outcome[violated_art_id] = 1
        
    aux_df = pd.DataFrame({'case_id':case_id,
                           'article_id': art_id,
                           'broken': outcome})
                           
    str_data_test = pd.concat([str_data_test, aux_df], axis = 0,
                             ignore_index=True)

str_data_test['dataset'] = ['test'] * len(str_data_test)

#%% Concatenate, reorganize and save dataframe

str_data_output = pd.concat([str_data_train, str_data_dev, str_data_test], axis = 0,
                            ignore_index=True)

print('shape final dataset = ', str_data_output.shape)

str_data_output.to_csv(output_path, index = False)

