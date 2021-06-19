#%% Imports
import pickle
import pandas as pd
from tqdm import tqdm

#%% Path definitions
path_weights = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/02_runs/batch_02/art_06/11_art6_50p_att_v4_50ep_TEST_DELETE/attn_weights.pkl'
path_test_dataset = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/01_preprocessed/01_article_split/art_06_50p_par/model_test.pkl'
path_tok_2_id = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/01_preprocessed/tok_2_id_dict.pkl'

#%% Data load
with open(path_weights, 'rb') as fr:
    weights = pickle.load(fr)
    
with open(path_test_dataset, 'rb') as fr:
    test_dataset = pickle.load(fr)
    
with open(path_tok_2_id, 'rb') as fr:
    tok_2_id = pickle.load(fr)
    
#%% Global initialization
idx_case = 0
n_art_pars = 11
n_case_pass = 50
seq_len = 512

id_2_tok = {}
print(f'Shape dataset before slicing = {test_dataset.shape}')
data = test_dataset[test_dataset.outcome == 1]
print(f'Shape dataset after slicing = {data.shape}')
weights_ECHR_art = weights['alpha_3']
weights_case_pass = weights['alpha_2']

#%% id_2_tok generation
for key in tok_2_id.keys():
    id_2_tok[tok_2_id[key]] = key

#%% Generate weight, ECHR_arts and case lists for specific entry
weight_ECHR_art = weights_ECHR_art[idx_case]
weight_case_pass = weights_case_pass[idx_case]

ECHR_pars = []
for idx in tqdm(range(n_art_pars)):
    beg = idx*seq_len
    end = (idx+1)*seq_len
    aux = data.iloc[idx_case]['article_pars_ids'][beg:end]
    aux = [id_2_tok[x] for x in aux]
    aux = [x for x in aux if x != '<PAD>']
    aux = (' ').join(aux)
    ECHR_pars.append(aux)

case_pass = []
for idx in tqdm(range(n_case_pass)):
    beg = idx*seq_len
    end = (idx+1)*seq_len
    aux = data.iloc[idx_case]['case_texts_ids'][beg:end]
    aux = [id_2_tok[x] for x in aux]
    aux = [x for x in aux if x != '<PAD>']
    aux = (' ').join(aux)
    case_pass.append(aux)

#%% Extract n most relevant paragraphs from article
ECHR_art_df = pd.DataFrame({'text': ECHR_pars,
                            'score': weight_ECHR_art})
ECHR_art_df_sorted = ECHR_art_df.sort_values(by=['score'], ascending=False)


#%% Extract n most relevant passages from case
case_pass_df = pd.DataFrame({'text': case_pass,
                             'score': weight_case_pass})
case_pass_df_sorted = case_pass_df.sort_values(by=['score'], ascending=False)

print(case_pass_df)

#%% Print 5 most relevant passages

case_pass_relevant = case_pass_df_sorted[case_pass_df_sorted['text'] != '']
case_pass_relevant = case_pass_relevant['text'][0:3]

for text in case_pass_relevant:
    print(text)







