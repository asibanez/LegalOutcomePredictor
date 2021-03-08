# v1 -> Extracts article paragraphs

#%% Imports

import pickle
from pprint import pprint

#%% Path definition

path_data = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/01_preprocessed/01_article_split/art_06_50p_par/model_train.pkl'
path_tok_2_id = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/01_preprocessed/tok_2_id_dict.pkl'

#%% Data loading

with open(path_data, 'rb') as fr:
    data = pickle.load(fr)

with open(path_tok_2_id, 'rb') as fr:
    tok_2_id = pickle.load(fr)

#%% Global initialization

pos = 0
n_case_texts = 300
n_art_pars = 11
len_case_texts = 512
id_2_tok = {}
art_par_strings = []
case_strings_all = {}

#%% tok_2_id generation

for key in tok_2_id.keys():
    id_2_tok[tok_2_id[key]] = key
    
#%% Article extraction

case_id = data.iloc[pos]['case_id']
art_id = data.iloc[pos]['article_id']
art_par_ids = data.iloc[pos]['article_pars_ids']
art_par_tok = [id_2_tok[x] for x in art_par_ids]

#%%
for idx in range(0, n_art_pars):
    aux = art_par_tok[idx*len_case_texts : (idx + 1)*len_case_texts]
    aux = [x for x in aux if x != '<PAD>']
    aux = (' ').join(aux)
    art_par_strings.append(aux)
for x in art_par_strings: print(x,'\n')

#%% Case extraction

case_ids_all = data.iloc[pos]['case_texts_ids']
case_tok_all = [id_2_tok[x] for x in case_ids_all]

#%%
for idx in range(0, n_case_texts):
    case_tok = case_tok_all[idx*len_case_texts : (idx+1)*len_case_texts]
    case_strings_all[idx] = (' ').join([x for x in case_tok if x != '<PAD>'])
    #case_strings_all[idx] = (' ').join([x for x in case_tok])

#%% Print results

print(f'CASE ID:\n{case_id}\n')
print(f'ARTICLE PARAGRAPHS:\n{art_par_strings}\n')
_ = [print(f'CASE_TEXT {x}\n{case_strings_all[x]}\n') for x in range(0,n_case_texts)]
