#%% Imports
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

#%% Path definition
input_folder = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/v2/01_preprocessed/04_toy_4'
#input_folder = '/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/v2/01_preprocessed/00_full'

train_set_path = os.path.join(input_folder, 'model_train.pkl')
val_set_path = os.path.join(input_folder, 'model_dev.pkl')
test_set_path = os.path.join(input_folder, 'model_test.pkl')

#%% Global initialization
pos = 0
max_num_pars = 200
seq_len = 512
par_start = 83
par_end = 84

#%% Data load
dataset = pd.read_pickle(train_set_path)

#%% Tokenizer instantiation
model_name = 'nlpaueb/legal-bert-small-uncased'
bert_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)

#%% Process entries
all_token_ids = dataset.iloc[pos]['facts_ids']

for idx in tqdm(range(par_start, par_end)):
    pos_b = seq_len*idx
    pos_e = seq_len*(idx + 1)
    par_token_ids = all_token_ids[pos_b:pos_e]
    par_text = bert_tokenizer.decode(par_token_ids)
    print(f'\nPar number: {idx}\n{par_text}\n')

#%% Various tests
aux = bert_tokenizer()
aux = bert_tokenizer('hello, how are you?', truncation=True, max_length=20, padding='max_length')
aux = bert_tokenizer('', truncation=True, max_length=20, padding='max_length')
bert_tokenizer.decode(aux['input_ids'])

