#%% Imports

import pickle

#%% Path definition

path_data = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/01_data/02_runs/12_art_6_300_pass/model_train.pkl'
path_tok_2_id = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/01_data/01_preprocessed/tok_2_id_dict.pkl'

#%% Data loading

with open(path_data, 'rb') as fr:
    data = pickle.load(fr)

with open(path_tok_2_id, 'rb') as fr:
    tok_2_id = pickle.load(fr)

#%% Global initialization

pos = 1
n_case_texts = 300
len_case_texts = 512
id_2_tok = {}
case_strings_all = {}

#%% tok_2_id generation

for key in tok_2_id.keys():
    id_2_tok[tok_2_id[key]] = key
    
#%% Article extraction

case_id = data.iloc[pos]['case_id']
article_id = data.iloc[pos]['article_text']
article_tok = [id_2_tok[x] for x in article_id]
article_string = (' ').join([x for x in article_tok if x != '<PAD>'])

#%% Case extraction

case_ids_all = data.iloc[pos]['case_texts']
case_tok_all = [id_2_tok[x] for x in case_ids_all]

for idx in range(0, n_case_texts):
    case_tok = case_tok_all[idx*len_case_texts : (idx+1)*len_case_texts]
    #case_strings_all[idx] = (' ').join([x for x in case_tok if x != '<PAD>'])
    case_strings_all[idx] = (' ').join([x for x in case_tok])

#%% Print results

print(f'CASE ID:\n{case_id}\n')
print(f'ARTICLE TEXT:\n{article_string}\n')
_ = [print(f'CASE_TEXT {x}\n{case_strings_all[x]}\n') for x in range(0,n_case_texts)]
