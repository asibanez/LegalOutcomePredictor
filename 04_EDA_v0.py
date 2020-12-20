#%% Imports
import os
import nltk
import tqdm
import codecs
import pickle
import pandas as pd
import matplotlib.pyplot as plt

#%% Path definition

base_folder = os.getcwd()
input_folder = os.path.join(base_folder, '01_data', '01_preprocessed')
output_path_= os.path.join(base_folder, '01_data', '01_preprocessed', 'corpus.txt')

train_dev_test_files = ['case_EN_train', 'case_EN_dev', 'case_EN_test']
ECHR_dict_path = os.path.join(input_folder, 'ECHR.dict')
case_train_path = os.path.join(input_folder, 'case_EN_train')
case_dev_path = os.path.join(input_folder, 'case_EN_dev')
case_test_path = os.path.join(input_folder, 'case_EN_test')

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
case_all_df = pd.concat([case_train_df, case_dev_df, case_test_df], axis = 0)
print('shape case all = ', case_all_df.shape)

#%% EDA cases

num_cases = len(case_all_df)
case_passages = case_all_df.TEXT.to_list()
case_text = [(' ').join(x) for x in  case_passages]
case_tokens = [nltk.word_tokenize(x) for x in case_text]

num_passages_in_case = [len(x) for x in case_passages]
num_tokens_in_case = [len(x) for x in case_tokens]

print('max num passages in case = ', max(num_passages_in_case))
print('max num tokens in case = ', max(num_tokens_in_case))
print('avg num passages in case = ', sum(num_passages_in_case)/num_cases)
print('avg num tokens in case = ', sum(num_tokens_in_case)/num_cases)

fig = plt.figure()
plt.hist(num_passages_in_case)
plt.xlabel('num passages in case')
plt.ylabel('freq')
plt.show()

fig = plt.figure()
plt.hist(num_tokens_in_case)
plt.xlabel('num tokens in case')
plt.ylabel('freq')
plt.show()

#%% EDA ECHR

num_echr_arts = len(ECHR_dict)
echr_art_text = list(ECHR_dict.values())
echr_art_tokens = [nltk.word_tokenize(x) for x in echr_art_text]
num_tokens_in_art = [len(x) for x in echr_art_tokens]

print('max num tokens in ECHR arts = ', max(num_tokens_in_art))
print('avg num tokens in ECHR arts = ', sum(num_tokens_in_art)/num_echr_arts)

fig = plt.figure()
plt.hist(num_tokens_in_art)
plt.xlabel('num tokens in echr art')
plt.ylabel('freq')
plt.show()
#%%
