# Selects paragraphs from case descriptions
# Tokenizes, prunes and pads sequences for article and paragraph texts
#-----------------------------------------------
# Input:  train, dev and test dataframes in pkl format
#         token to ID dictionary in pkl format

# Output: train, dev and test dataframes as model inputs:
#-----------------------------------------------
# v1 ->  Generates tokenized text
# v2 ->  Saves dataframes in pkl format
# v3 ->  Bug fixed line 19: Pads top n articles to num_passages per case
# v5 ->  Adds oversampling to train dataset
#        Uses pd.read_csv instead of pickle.load()
#        Adds random seed
# v6 ->  Adds conversion to IDs, sequence prunning and padding
# v7 ->  Saves case texts as flat token ID lists
# v8 ->  Adds bm25 paragraph selection option and moves passage processing to
#        select_process_passage function
# v9 ->  Fixed bug: adds empty passage when n_passages < 5
# v10 -> Allows to choose the ECHR articles to include in the model
# v11 -> Preprocessing moved to function
#        Added option to include unaltered passage sequence
# v12 -> Case ID added to output

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
# Paragraph selection
def select_process_passage(num_passages_per_case, case_texts, seq_len, pad_token_id,
                           art_text_tok_list, par_selection_method):

    selected_passages_ids_list = []
    
    for art_text_tok in art_text_tok_list:
        
        # Select passages
        if par_selection_method == 'random':
            selected_passages = random.sample(case_texts, min(num_passages_per_case, len(case_texts)))
        
        elif par_selection_method == 'bm25':
            tokenized_corpus = [doc.split(' ') for doc in case_texts]
            bm25 = BM25Okapi(tokenized_corpus)
            selected_passages = bm25.get_top_n(art_text_tok, case_texts, n = num_passages_per_case)    
        
        elif par_selection_method == 'none':
            selected_passages = case_texts[0:num_passages_per_case]            
        
        else:
            print('Error: Incorrect paragraph selection method')
            exit()
            
        # Convert to lowercase and tokenize passages
        selected_passages_tok = [nltk.word_tokenize(x.lower()) for x in selected_passages]
        # Convert to IDs
        selected_passages_ids = [[tok_2_id[token] for token in text] for text in selected_passages_tok]
        # Prune to seq len
        selected_passages_ids = [x[:seq_len] for x in selected_passages_ids] 
        # Pad to seq len
        selected_passages_ids = [x + [pad_token_id] * (seq_len - len(x)) for x in selected_passages_ids]
        # Pad number of passages to desired length
        selected_passages_ids += [[0] * seq_len] * (num_passages_per_case - len(selected_passages_ids)) 
        # Flatten list of lists
        selected_passages_ids = [x for sublist in selected_passages_ids for x in sublist]

        # build list    
        selected_passages_ids_list.append(selected_passages_ids)

    return selected_passages_ids_list

#%% Data restructuring and conversion to IDs

def dataset_preproc_f(dataset_df, article_list, num_passages_per_case, seq_len,
                         pad_token_id, art_text_tok_list, par_selection_method):

    str_data = pd.DataFrame(columns = ['case_id', 'article_id', 'article_text',
                                       'case_texts', 'outcome'])
    
    # Iterate over cases to build dataframe
    for case in tqdm.tqdm(dataset_df.iterrows(), total = len(dataset_df)):
        case_id = case[1]['ITEMID']
        case_texts = case[1]['TEXT']
        violated_art_ids = case[1]['VIOLATED_ARTICLES']
        # Initialize outcomes and case ID lists
        case_id_list = [case_id] * len(article_list)
        outcome_list = [0] * len(article_list)
        # Select paragraphs from case and build list
        selected_case_passages_id_list = select_process_passage(num_passages_per_case,
                                                                case_texts, seq_len,
                                                                pad_token_id,
                                                                art_text_tok_list,
                                                                par_selection_method)
        
        # Iterate over violated article list
        for violated_art_id in violated_art_ids:
            # Skip unwanted articles
            if violated_art_id in arts_to_skip:
                continue
            violated_art_id = int(violated_art_id)
            # Update corresponding outcome label
            for idx, article in enumerate(article_list):
                if violated_art_id == article:
                    outcome_list[idx] = 1
            
        aux_df = pd.DataFrame({'case_id': case_id_list,
                               'article_id': article_list,
                               'article_text':art_text_id_list,
                               'case_texts': selected_case_passages_id_list,
                               'outcome': outcome_list})
                               
        str_data = pd.concat([str_data, aux_df], axis = 0, ignore_index = True)
        
    return(str_data)

#%% Path definition

base_folder = os.path.split(os.getcwd())[0]
input_folder = os.path.join(base_folder, '01_data', '01_preprocessed')
output_folder =  os.path.join(base_folder, '01_data', '02_runs', '13_art_6_50_pass')
output_path_train = os.path.join(output_folder, 'model_train.pkl')
output_path_dev = os.path.join(output_folder, 'model_dev.pkl')
output_path_test = os.path.join(output_folder, 'model_test.pkl')

ECHR_dict_path = os.path.join(input_folder, 'ECHR_dict.pkl')
case_train_path = os.path.join(input_folder, 'case_EN_train_df.pkl')
case_dev_path = os.path.join(input_folder, 'case_EN_dev_df.pkl')
case_test_path = os.path.join(input_folder, 'case_EN_test_df.pkl')
tok_2_id_path = os.path.join(input_folder, 'tok_2_id_dict.pkl')

#%% Variable initialization

num_articles = 66 # total number of articles in ECHR law
num_passages_per_case = 50 # Number of case paragraphs to be fed to the model
par_selection_method = 'none' # Either 'bm25', 'random' or 'none'
article_list = [3] #[4, 5, 6, 7]
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
art_text_list = [ECHR_dict[x] for x in article_list]
art_text_tok_list = [nltk.word_tokenize(x.lower()) for x in art_text_list]
art_text_id_list = [[tok_2_id[token] for token in text] for text in art_text_tok_list]

# Prune to seq len
art_text_id_list = [x[:seq_len] for x in art_text_id_list]
       
# Pad to seq len
art_text_id_list = [x + [pad_token_id] * (seq_len - len(x)) for x in art_text_id_list]


#%% Restructure train / dev / test datasets and convert tokens to ids
                           
str_data_train = dataset_preproc_f(case_train_df, article_list, num_passages_per_case,
                                   seq_len, pad_token_id, art_text_tok_list, par_selection_method)

str_data_dev = dataset_preproc_f(case_dev_df, article_list, num_passages_per_case,
                                 seq_len, pad_token_id, art_text_tok_list, par_selection_method)

str_data_test = dataset_preproc_f(case_test_df, article_list, num_passages_per_case,
                                  seq_len, pad_token_id, art_text_tok_list, par_selection_method)

#%% Oversample train data

x_train = pd.DataFrame(str_data_train[['case_id', 'article_id', 'article_text', 'case_texts']])
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
