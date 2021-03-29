# v1 -> Extracts article paragraphs

#%% Imports

import pickle

#%% Path definition

path_data = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/01_preprocessed/01_article_split/art_06_50p_par_att/model_train.pkl'
path_tok_2_id = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/01_preprocessed/tok_2_id_dict.pkl'

#%% Data loading

with open(path_data, 'rb') as fr:
    data = pickle.load(fr)

with open(path_tok_2_id, 'rb') as fr:
    tok_2_id = pickle.load(fr)

#%% Global initialization

pos = 10259
n_case_texts = 50
n_art_pars = 11
seq_len = 512
id_2_tok = {}
art_pars_strings = []
case_strings = []

#%% tok_2_id generation

for key in tok_2_id.keys():
    id_2_tok[tok_2_id[key]] = key
    
#%% Extract from dataframe

case_id = data.iloc[pos]['case_id']
ECHR_art_id = data.iloc[pos]['article_id']
art_pars_ids = data.iloc[pos]['article_pars_ids']
case_texts_ids = data.iloc[pos]['case_texts_ids']
violated_pars_ids = data.iloc[pos]['violated_pars']

#%% Convert ids to tokens
art_pars_tok = [id_2_tok[x] for x in art_pars_ids]
case_texts_tok = [id_2_tok[x] for x in case_texts_ids]

#%% Article extraction
for idx in range(0, n_art_pars):
    aux = art_pars_tok[idx * seq_len : (idx + 1) * seq_len]
    aux = [x for x in aux if x != '<PAD>']
    aux = (' ').join(aux)
    if aux != '':
        art_pars_strings.append(aux)
    
#%% Case extraction
for idx in range(0, n_case_texts):
    aux = case_texts_tok[idx * seq_len : (idx+1) * seq_len]
    aux = [x for x in aux if x != '<PAD>']
    aux = (' ').join(aux)
    if aux != '':
        case_strings.append(aux)
    
#%% Print example

print(f'CASE ID: {case_id}\n')
print(f'ECHR art id: {ECHR_art_id}')
_ = [print(f'ART_PARAGRAPH {idx}\n{text}\n') for idx, text in enumerate(art_pars_strings)]
_ = [print(f'CASE_TEXT {idx}\n{text}\n') for idx, text in enumerate(case_strings)]
