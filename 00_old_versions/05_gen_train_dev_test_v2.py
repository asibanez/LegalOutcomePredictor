# v1 -> Generates tokenized text
# v2 -> Saves dataframes in pkl format
# v3 -> Bug fixed: Pads top n articles to num_passages per case

#%% Imports
import os
import tqdm
import nltk
import random
import pickle
import pandas as pd

#%% Function definitions
def select_passage(num_passages_per_case, case_texts):
    # Random selection
    selected_passages = random.sample(case_texts,
                                     min(num_passages_per_case, len(case_texts)))
    # Pad passages to desired length
    selected_passages += '' * (num_passages_per_case - len(selected_passages)) 
    
    # Convert to lowercase and tokenize
    selected_passages_tokens = [nltk.word_tokenize(x.lower()) for x in selected_passages]
    
    return selected_passages_tokens

#%% Path definition

base_folder = os.getcwd()
input_folder = os.path.join(base_folder, '01_data', '01_preprocessed')
output_path_train = os.path.join(base_folder, '01_data', '01_preprocessed', 'model_train.pkl')
output_path_dev = os.path.join(base_folder, '01_data', '01_preprocessed', 'model_dev.pkl')
output_path_test = os.path.join(base_folder, '01_data', '01_preprocessed', 'model_test.pkl')

train_dev_test_files = ['case_EN_train', 'case_EN_dev', 'case_EN_test']
ECHR_dict_path = os.path.join(input_folder, 'ECHR.dict')
case_train_path = os.path.join(input_folder, 'case_EN_train')
case_dev_path = os.path.join(input_folder, 'case_EN_dev')
case_test_path = os.path.join(input_folder, 'case_EN_test')

#%% Variable initializatoin

arts_to_skip = ['P1']
num_passages_per_case = 5

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

str_data_train = pd.DataFrame(columns = ['article_text', 'case_texts', 'outcome'])

for case in tqdm.tqdm(case_test_df.iterrows()):
    art_text = [ECHR_dict[x] for x in range(1,67)]
    art_text_tokens = [nltk.word_tokenize(x.lower()) for x in art_text]
    outcome = [0] * 66
    
    case_texts = case[1]['TEXT']
    violated_art_ids = case[1]['VIOLATED_ARTICLES']
    selected_case_texts_tokens = [select_passage(num_passages_per_case, case_texts)] * 66
    
    for violated_art_id in violated_art_ids:
        ### Skip unwanted articles
        if violated_art_id in arts_to_skip:
            continue
        violated_art_id = int(violated_art_id)
        outcome[violated_art_id] = 1
        
    aux_df = pd.DataFrame({'article_text':art_text_tokens,
                           'case_texts': selected_case_texts_tokens,
                           'outcome': outcome})
                           
    str_data_train = pd.concat([str_data_train, aux_df], axis = 0,
                               ignore_index=True)
    
#%% Restructure dev data

str_data_dev = pd.DataFrame(columns = ['article_text', 'case_texts', 'outcome'])

for case in tqdm.tqdm(case_test_df.iterrows()):
    art_text = [ECHR_dict[x] for x in range(1,67)]
    art_text_tokens = [nltk.word_tokenize(x.lower()) for x in art_text]
    outcome = [0] * 66
    
    case_texts = case[1]['TEXT']
    violated_art_ids = case[1]['VIOLATED_ARTICLES']
    selected_case_texts_tokens = [select_passage(num_passages_per_case, case_texts)] * 66
    
    for violated_art_id in violated_art_ids:
        ### Skip unwanted articles
        if violated_art_id in arts_to_skip:
            continue
        violated_art_id = int(violated_art_id)
        outcome[violated_art_id] = 1
        
    aux_df = pd.DataFrame({'article_text':art_text_tokens,
                           'case_texts': selected_case_texts_tokens,
                           'outcome': outcome})
                           
    str_data_dev = pd.concat([str_data_dev, aux_df], axis = 0,
                               ignore_index=True)  

#%% Restructure test data

str_data_test = pd.DataFrame(columns = ['article_text', 'case_texts', 'outcome'])

for case in tqdm.tqdm(case_test_df.iterrows()):
    art_text = [ECHR_dict[x] for x in range(1,67)]
    art_text_tokens = [nltk.word_tokenize(x.lower()) for x in art_text]
    outcome = [0] * 66
    
    case_texts = case[1]['TEXT']
    violated_art_ids = case[1]['VIOLATED_ARTICLES']
    selected_case_texts_tokens = [select_passage(num_passages_per_case, case_texts)] * 66
    
    for violated_art_id in violated_art_ids:
        ### Skip unwanted articles
        if violated_art_id in arts_to_skip:
            continue
        violated_art_id = int(violated_art_id)
        outcome[violated_art_id] = 1
        
    aux_df = pd.DataFrame({'article_text':art_text_tokens,
                           'case_texts': selected_case_texts_tokens,
                           'outcome': outcome})
                           
    str_data_test = pd.concat([str_data_test, aux_df], axis = 0,
                               ignore_index=True)
    
#%% Save train, dev, test dataframes

print('shape train = ', str_data_train.shape)
print('shape dev = ', str_data_dev.shape)
print('shape test = ', str_data_test.shape)

with open(output_path_train, 'wb') as fw:
    pickle.dump(str_data_train, fw)

with open(output_path_dev, 'wb') as fw:
    pickle.dump(str_data_dev, fw)

with open(output_path_test, 'wb') as fw:
    pickle.dump(str_data_test, fw)
