# Selects paragraphs from case descriptions
# Tokenizes, prunes and pads sequences for article and paragraph texts
#-----------------------------------------------
# Input:  train, dev and test dataframes in pkl format
#         token to ID dictionary in pkl format

# Output: train, dev and test dataframes as model inputs:
#-----------------------------------------------
# v1 -> Generates tokenized text
# v2 -> Saves dataframes in pkl format
# v3 -> Bug fixed line 19: Pads top n articles to num_passages per case
# v5 -> Adds oversampling to train dataset
#       Uses pd.read_csv instead of pickle.load()
#       Adds random seed
# v6 -> Adds conversion to IDs, sequence prunning and padding
# v7 -> Saves case texts as flat token ID lists
# v8 -> Adds bm25 paragraph selection option

#%% Imports
import os
import tqdm
import nltk
import random
import pickle
import pandas as pd
from rank_bm25 import BM25Okapi
from collections import defaultdict
from imblearn.over_sampling import RandomOverSampler 

#%% Function definitions
def select_passage(num_passages_per_case, case_texts, seq_len, pad_token_id,
                   art_texts_tokens, par_selection_method):

    selected_passages_ids_list = []
    
    if par_selection_method == 'random':
        #selected_passages = random.sample(case_texts,
        #                                  min(num_passages_per_case, len(case_texts)))
        pass
    
    elif par_selection_method == 'bm25':
        for art_text_tok in art_texts_tokens:
            tokenized_corpus = [doc.split(' ') for doc in case_texts]
            bm25 = BM25Okapi(tokenized_corpus)
            selected_passages = bm25.get_top_n(art_text_tok, case_texts, n = num_passages_per_case)    
            # Tokenize
            selected_passages_tok = [nltk.word_tokenize(x) for x in selected_passages]
            # Convert to IDs
            selected_passages_ids = [[tok_2_id[token] for token in text] for text in selected_passages_tok]
            # Prune to seq len
            selected_passages_ids = [x[:seq_len] for x in selected_passages_ids] 
            # Pad to seq len
            selected_passages_ids = [x + [pad_token_id] * (seq_len - len(x)) for x in selected_passages_ids]
            # Flatten list of lists
            selected_passages_ids = [x for sublist in selected_passages_ids for x in sublist]
            # build list    
            selected_passages_ids_list.append(selected_passages_ids)
        
        # Pad number of passages to desired length
        selected_passages_ids_list += [''] * (num_passages_per_case - len(selected_passages_ids_list)) 
    
    else:
        print('Error: Incorrect paragraph selection method')
        exit()
    
    # Convert to lowercase and tokenize
    selected_passages_tokens = [nltk.word_tokenize(x.lower()) for x in selected_passages]
    
    return selected_passages_tokens

#%% Path definition

base_folder = os.getcwd()
input_folder = os.path.join(base_folder, '01_data', '01_preprocessed')
output_path_train = os.path.join(base_folder, '01_data', '01_preprocessed', 'model_train.pkl')
output_path_dev = os.path.join(base_folder, '01_data', '01_preprocessed', 'model_dev.pkl')
output_path_test = os.path.join(base_folder, '01_data', '01_preprocessed', 'model_test.pkl')

#train_dev_test_files = ['case_EN_train', 'case_EN_dev', 'case_EN_test']
ECHR_dict_path = os.path.join(input_folder, 'ECHR_dict.pkl')
case_train_path = os.path.join(input_folder, 'case_EN_train_df.pkl')
case_dev_path = os.path.join(input_folder, 'case_EN_dev_df.pkl')
case_test_path = os.path.join(input_folder, 'case_EN_test_df.pkl')
tok_2_id_path = os.path.join(input_folder, 'tok_2_id_dict.pkl')

#%% Variable initialization

par_selection_method = 'bm25' # Either 'bm25' or 'random'
num_passages_per_case = 5 # Number of case paragraphs to be fed to the model
num_articles = 67 # total number of articles in ECHR law
arts_to_skip = ['P1', 'P4', 'P7', 'P12', 'P6'] # Art headers to be skipped
pad_token_id = 0
seq_len = 512
seed = 1234

#%% Load data

with open(ECHR_dict_path, 'rb') as fr:
    ECHR_dict = pickle.load(fr)
    
with open(tok_2_id_path, 'rb') as fr:
    tok_2_id = pickle.load(fr) 

case_train_df = pd.read_pickle(case_train_path)
case_dev_df = pd.read_pickle(case_dev_path)
case_test_df = pd.read_pickle(case_test_path)

#%% Convert tok_2_id to default dictionary

unk_id = tok_2_id['<UNK>']
tok_2_id = defaultdict(lambda: unk_id, tok_2_id)

#%% Asess dataframe shapes

print('shape case train = ', case_train_df.shape)
print('shape case dev = ', case_dev_df.shape)
print('shape case test = ', case_test_df.shape)

#%% Convert articles to token IDs, prune and / or pad to seq len

# Convert to IDs
art_texts = [ECHR_dict[x] for x in range(1, num_articles)]
art_texts_tokens = [nltk.word_tokenize(x.lower()) for x in art_texts]
art_texts_ids = [[tok_2_id[token] for token in text] for text in art_texts_tokens]

# Prune to seq len
art_texts_ids = [x[:seq_len] for x in art_texts_ids]
       
# Pad to seq len
art_texts_ids = [x + [pad_token_id] * (seq_len - len(x)) for x in art_texts_ids]


#%% Restructure train data and convert tokens to ids

str_data_train = pd.DataFrame(columns = ['article_text', 'case_texts', 'outcome'])

# Iterate over cases to build dataframe
for case in tqdm.tqdm(case_train_df.iterrows(), total = len(case_train_df)):
    case_texts = case[1]['TEXT']
    violated_art_ids = case[1]['VIOLATED_ARTICLES']
    # Initialize outcomes to zero
    outcome = [0] * (num_articles - 1)
    # Select paragraphs from case and build list
    selected_case_text_tokens = select_passage(num_passages_per_case, case_texts, seq_len,
                                               pad_token_id, art_texts_tokens, par_selection_method)
    # Convert to IDs
    selected_case_text_ids = [[tok_2_id[token] for token in text] for text in selected_case_text_tokens]
    # Prune to seq len
    selected_case_text_ids = [x[:seq_len] for x in selected_case_text_ids]       
    # Pad to seq len
    selected_case_text_ids = [x + [pad_token_id] * (seq_len - len(x)) for x in selected_case_text_ids]
    # Flatten list of lists
    selected_case_text_ids = [x for sublist in selected_case_text_ids for x in sublist]
    # build list    
    selected_case_texts_ids = [selected_case_text_ids] * 66
    
    # Iterate over violated article list
    for violated_art_id in violated_art_ids:
        # Skip unwanted articles
        if violated_art_id in arts_to_skip:
            continue
        violated_art_id = int(violated_art_id)
        # Update corresponding outcome label
        outcome[violated_art_id] = 1
        
    aux_df = pd.DataFrame({'article_text':art_texts_ids,
                           'case_texts': selected_case_texts_ids,
                           'outcome': outcome})
                           
    str_data_train = pd.concat([str_data_train, aux_df], axis = 0,
                               ignore_index = True)

#%% Restructure dev data and convert tokens to ids

str_data_dev = pd.DataFrame(columns = ['article_text', 'case_texts', 'outcome'])

# Iterate over cases to build dataframe
for case in tqdm.tqdm(case_dev_df.iterrows(), total = len(case_dev_df)):
    case_texts = case[1]['TEXT']
    violated_art_ids = case[1]['VIOLATED_ARTICLES']
    # Initialize outcomes to zero
    outcome = [0] * (num_articles - 1)
    # Select paragraphs from case and build list
    selected_case_text_tokens = select_passage(num_passages_per_case, case_texts)
    # Convert to IDs
    selected_case_text_ids = [[tok_2_id[token] for token in text] for text in selected_case_text_tokens]
    # Prune to seq len
    selected_case_text_ids = [x[:seq_len] for x in selected_case_text_ids]       
    # Pad to seq len
    selected_case_text_ids = [x + [pad_token_id] * (seq_len - len(x)) for x in selected_case_text_ids]
    # Flatten list of lists
    selected_case_text_ids = [x for sublist in selected_case_text_ids for x in sublist]
    # build list    
    selected_case_texts_ids = [selected_case_text_ids] * 66
    
    # Iterate over violated article list
    for violated_art_id in violated_art_ids:
        # Skip unwanted articles
        if violated_art_id in arts_to_skip:
            continue
        violated_art_id = int(violated_art_id)
        # Update corresponding outcome label
        outcome[violated_art_id] = 1
        
    aux_df = pd.DataFrame({'article_text':art_texts_ids,
                           'case_texts': selected_case_texts_ids,
                           'outcome': outcome})
                           
    str_data_dev = pd.concat([str_data_dev, aux_df], axis = 0,
                               ignore_index = True)

#%% Restructure test data and convert tokens to ids

str_data_test = pd.DataFrame(columns = ['article_text', 'case_texts', 'outcome'])

# Iterate over cases to build dataframe
for case in tqdm.tqdm(case_test_df.iterrows(), total = len(case_test_df)):
    case_texts = case[1]['TEXT']
    violated_art_ids = case[1]['VIOLATED_ARTICLES']
    # Initialize outcomes to zero
    outcome = [0] * (num_articles - 1)
    # Select paragraphs from case and build list
    selected_case_text_tokens = select_passage(num_passages_per_case, case_texts)
    # Convert to IDs
    selected_case_text_ids = [[tok_2_id[token] for token in text] for text in selected_case_text_tokens]
    # Prune to seq len
    selected_case_text_ids = [x[:seq_len] for x in selected_case_text_ids]       
    # Pad to seq len
    selected_case_text_ids = [x + [pad_token_id] * (seq_len - len(x)) for x in selected_case_text_ids]
    # Flatten list of lists
    selected_case_text_ids = [x for sublist in selected_case_text_ids for x in sublist]
    # build list    
    selected_case_texts_ids = [selected_case_text_ids] * 66
    
    # Iterate over violated article list
    for violated_art_id in violated_art_ids:
        # Skip unwanted articles
        if violated_art_id in arts_to_skip:
            continue
        violated_art_id = int(violated_art_id)
        # Update corresponding outcome label
        outcome[violated_art_id] = 1
        
    aux_df = pd.DataFrame({'article_text':art_texts_ids,
                           'case_texts': selected_case_texts_ids,
                           'outcome': outcome})
                           
    str_data_test = pd.concat([str_data_test, aux_df], axis = 0,
                               ignore_index = True)

#%% Oversample train data

x_train = pd.DataFrame(str_data_train[['article_text', 'case_texts']])
y_train = pd.DataFrame(str_data_train['outcome']).astype('int')

random_oversampler = RandomOverSampler(random_state = seed)
x_train_res, y_train_res = random_oversampler.fit_resample(x_train, y_train)

str_data_train = pd.concat([x_train_res, y_train_res], axis = 1)
       
#%% Save train, dev, test dataframes

print('shape train = ', str_data_train.shape)
print('shape dev = ', str_data_dev.shape)
print('shape test = ', str_data_test.shape)

str_data_train.to_pickle(output_path_train)
str_data_dev.to_pickle(output_path_dev)
str_data_test.to_pickle(output_path_test)
