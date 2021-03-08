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
# v13 -> Uses articles paragraphs

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

def dataset_preproc_f(dataset_df, ECHR_art_df, ECHR_par_df, selected_arts,
                      num_passages_per_case, seq_len, max_num_pars, 
                      pad_token_id, par_selection_method):

    str_data = pd.DataFrame(columns = ['case_id', 'article_id', 'article_pars_ids',
                                       'case_texts_ids', 'outcome'])
    
    # Iterate over cases to build dataframe
    for _, case in tqdm.tqdm(dataset_df.iterrows(), total = len(dataset_df)):
        case_id = case['ITEMID']
        case_texts = case['TEXT']
        violated_art_ids = case['VIOLATED_ARTICLES']
        
        art_text_tok_list = list(ECHR_art_df.loc[selected_arts]['art_tok'])
        
        # Initialize outcomes and case ID list
        art_pars_id_list = []
        case_id_list = [case_id] * len(selected_arts)
        outcome_list = [0] * len(selected_arts)
 
        # Extract article paragrpahs
        for ECHR_art in selected_arts:
            # Extract paragraph IDs
            aux = list(ECHR_par_df[ECHR_par_df['Art ID'] == ECHR_art]['par_ids'])
            # Flatten article paragraphs
            aux = [x for sublist in aux for x in sublist]
            # Pad to max number of paragraphs
            aux = aux + [pad_token_id] * (max_num_pars * seq_len - len(aux))
            # Append to article paragraphs id list
            art_pars_id_list.append(aux)       
 
        # Select paragraphs from case and build list
        selected_case_passages_id_list = select_process_passage(num_passages_per_case,
                                                                case_texts, seq_len,
                                                                pad_token_id,
                                                                art_text_tok_list,
                                                                par_selection_method)
        
        # Iterate over violated article list
        for idx, violated_art_id in enumerate(violated_art_ids):
                       
            # Update corresponding outcome label
            for idx, article in enumerate(selected_arts):
                if violated_art_id == article:
                    outcome_list[idx] = 1
            
        aux_df = pd.DataFrame({'case_id': case_id_list,
                               'article_id': selected_arts,
                               'article_pars_ids': art_pars_id_list,
                               'case_texts_ids': selected_case_passages_id_list,
                               'outcome': outcome_list})
                               
        str_data = pd.concat([str_data, aux_df], axis = 0, ignore_index = True)
        
    return(str_data)

#%% Path definition

#input_folder = 'C://Users//siban//Dropbox//CSAIL//Projects//12_Legal_Outcome_Predictor//00_data//01_preprocessed'
#output_base_folder = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/01_preprocessed/01_article_split'

input_folder = '/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/01_preprocessed'
output_base_folder = '/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/01_preprocessed/01_article_split'

output_folder =  os.path.join(output_base_folder, 'art_06_50p_par')
output_path_train = os.path.join(output_folder, 'model_train.pkl')
output_path_dev = os.path.join(output_folder, 'model_dev.pkl')
output_path_test = os.path.join(output_folder, 'model_test.pkl')

ECHR_path = os.path.join(input_folder, 'ECHR_paragraphs.csv')
case_train_path = os.path.join(input_folder, 'case_EN_train_df.pkl')
case_dev_path = os.path.join(input_folder, 'case_EN_dev_df.pkl')
case_test_path = os.path.join(input_folder, 'case_EN_test_df.pkl')
tok_2_id_path = os.path.join(input_folder, 'tok_2_id_dict.pkl')

if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
        print("Created folder : ", output_folder)

#%% Global initialization

num_passages_per_case = 50 # Number of case paragraphs to be fed to the model
par_selection_method = 'bm25' # Either 'bm25', 'random' or 'none'
selected_arts = ['6'] #[x for x in range(1, 67)] #[10] #[4, 5, 6, 7]
pad_token_id = 0
seq_len = 512
max_num_pars = 11
seed = 1234

#%% Load data
  
with open(tok_2_id_path, 'rb') as fr:
    tok_2_id = pickle.load(fr) 

case_train_df = pd.read_pickle(case_train_path)
case_dev_df = pd.read_pickle(case_dev_path)
case_test_df = pd.read_pickle(case_test_path)
ECHR_par_df = pd.read_csv(ECHR_path)

#%% Convert tok_2_id to default dictionary

unk_id = tok_2_id['<UNK>']
tok_2_id = defaultdict(lambda: unk_id, tok_2_id)

#%% Asess dataframe shapes

print('shape case train = ', case_train_df.shape)
print('shape case dev = ', case_dev_df.shape)
print('shape case test = ', case_test_df.shape)
print('shape ECHR = ', ECHR_par_df.shape)

#%% Convert ECHR paragraphs to token IDs, prune and / or pad to seq len

# Convert to IDs
ECHR_par_list = list(ECHR_par_df['Text'])
ECHR_par_tok_list = [nltk.word_tokenize(x.lower()) for x in ECHR_par_list]
ECHR_par_id_list = [[tok_2_id[token] for token in par] for par in ECHR_par_tok_list]

# Prune to seq len
ECHR_par_id_list = [x[:seq_len] for x in ECHR_par_id_list]

# Pad to seq len
ECHR_par_id_list = [x + [pad_token_id] * (seq_len - len(x)) for x in ECHR_par_id_list]

ECHR_par_df['par_ids'] = ECHR_par_id_list

#%% Generate article text from article paragraphs and tokenize

ECHR_art_df = pd.DataFrame(ECHR_par_df.groupby(['Art ID'])['Text'].apply(lambda x: ' '.join(x)))
ECHR_art_list = list(ECHR_art_df['Text'])
ECHR_art_tok_list = [nltk.word_tokenize(x.lower()) for x in ECHR_art_list]

ECHR_art_df['art_tok'] = ECHR_art_tok_list


#%% Restructure train / dev / test datasets and convert tokens to ids
                     
str_data_train = dataset_preproc_f(case_train_df, ECHR_art_df, ECHR_par_df, selected_arts,
                                   num_passages_per_case, seq_len, max_num_pars,
                                   pad_token_id, par_selection_method)

str_data_dev = dataset_preproc_f(case_dev_df, ECHR_art_df, ECHR_par_df, selected_arts,
                                 num_passages_per_case, seq_len, max_num_pars,
                                 pad_token_id, par_selection_method)

str_data_test = dataset_preproc_f(case_test_df, ECHR_art_df, ECHR_par_df, selected_arts,
                                  num_passages_per_case, seq_len, max_num_pars,
                                  pad_token_id, par_selection_method)

#%% Oversample train data

x_train = pd.DataFrame(str_data_train[['case_id', 'article_id', 'article_pars_ids', 'case_texts_ids']])
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
