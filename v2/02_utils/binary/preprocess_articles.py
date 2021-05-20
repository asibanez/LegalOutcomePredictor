#%% Imports
import pandas as pd

#%% Path definition
path_ECHR = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/v2/00_ECHR_law/01_parsed/ECHR_paragraphs_final_no_bullets.csv'

#%% Global initialization
num_labels = 33
seq_len = 256
max_n_pars = 50
pad_int = 0
toy_data = False
len_toy_data = 100
padded_len = seq_len * max_n_pars
id_2_label = {0: '2',
              1: '3',
              2: '4',
              3: '5',
              4: '6',
              5: '7',
              6: '8',
              7: '9',
              8: '10',
              9: '11',
              10:'12',
              11:'13',
              12:'14',
              13:'15',
              14:'17',
              15:'18',
              16:'34',
              17:'38',
              18:'39',
              19:'46',              
              20:'P1-1',
              21:'P1-2',
              22:'P1-3',
              23:'P3-1',
              24:'P4-2',
              25:'P4-4',
              26:'P6-3',
              27:'P7-1',
              28:'P7-2',
              29:'P7-3',
              30:'P7-4',
              31:'P7-5',
              32:'P12-1'}

#%% Data loading
data = pd.read_csv(path_ECHR)

#%% Assess all required articles in dataset
ids_in_dict = set(list(id_2_label.values()))
ids_in_dataset = set(list(data['Art ID']))
missing = ids_in_dict.difference(ids_in_dataset)
print(f'Missing articles in dataset = {missing}')

#%% Convert ECHR dataset 

new = pd.DataFrame(data.groupby(['Art ID'])['Text'].apply(list)).reset_index()




