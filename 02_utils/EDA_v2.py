# v1 -> Includes computations of broken article distribution
# v2 -> Minor improvements

#%% Imports
import os
import nltk
import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#%% Path definition

base_folder = os.getcwd()
input_folder = os.path.join(base_folder, '01_data', '01_preprocessed')

ECHR_dict_path = os.path.join(input_folder, 'ECHR_dict.pkl')
case_train_path = os.path.join(input_folder, 'case_EN_train_df.pkl')
case_dev_path = os.path.join(input_folder, 'case_EN_dev_df.pkl')
case_test_path = os.path.join(input_folder, 'case_EN_test_df.pkl')

#%% Load data

with open(ECHR_dict_path, 'rb') as fr:
    ECHR_dict = pickle.load(fr)

case_train_df = pd.read_pickle(case_train_path)
case_dev_df = pd.read_pickle(case_dev_path)
case_test_df = pd.read_pickle(case_test_path)

#%% Merge case dataframes

print('shape case train = ', case_train_df.shape)
print('shape case dev = ', case_dev_df.shape)
print('shape case test = ', case_test_df.shape)

case_all_df = pd.concat([case_train_df, case_dev_df, case_test_df], axis = 0)
print('shape case all = ', case_all_df.shape)

#%% Compute num tokens and num passages per caseEDA cases

num_cases = len(case_all_df)
cases_passages = case_all_df.TEXT.to_list()
cases_passages_tok = [[nltk.word_tokenize(x) for x in case_passages] for case_passages in cases_passages]
all_passages_tok = [x for sublist in cases_passages_tok for x in sublist]
cases_text_tok = [[x for sublist in passages for x in sublist] for passages in cases_passages_tok]

num_passages = len(all_passages_tok)
num_passages_per_case = [len(x) for x in cases_passages_tok]
num_tokens_per_passage = [len(x) for x in all_passages_tok]
num_tokens_per_case = [len(x) for x in cases_text_tok]

print('max num passages in case = ', max(num_passages_per_case))
print('avg num passages in case = ', sum(num_passages_per_case)/num_cases)

print('max num tokens in passage = ', max(num_tokens_per_passage))
print('avg num tokens in passage = ', sum(num_tokens_per_passage)/num_passages)

print('max num tokens in case = ', max(num_tokens_per_case))
print('avg num tokens in case = ', sum(num_tokens_per_case)/num_cases)

#%% Plot num tokens and num passages per case

fig = plt.figure()
plt.hist(num_passages_per_case, bins = 50)
plt.xlabel('num passages in case')
plt.ylabel('freq')
plt.xlim(0, 500)
plt.show()

fig = plt.figure()
plt.hist(num_tokens_per_passage, bins = 50)
plt.xlabel('num tokens in passage')
plt.ylabel('freq')
plt.xlim(0, 1000)
plt.show()

fig = plt.figure()
plt.hist(num_tokens_per_case, bins = 50)
plt.xlabel('num tokens in case')
plt.xlim(0, 30000)
plt.ylabel('freq')
plt.show()

#%% Compute distribution of broken articles
   
violated_art_train = [x for sublist in list(case_train_df.VIOLATED_ARTICLES) for x in sublist]
violated_art_train = [x for x in violated_art_train if 'P' not in x]
violated_art_train_count = pd.value_counts(violated_art_train)
violated_art_train_int = [int(x) for x in violated_art_train]

violated_art_dev = [x for sublist in list(case_dev_df.VIOLATED_ARTICLES) for x in sublist]
violated_art_dev = [x for x in violated_art_dev if 'P' not in x]
violated_art_dev_count = pd.value_counts(violated_art_dev)
violated_art_dev_int = [int(x) for x in violated_art_dev]

violated_art_test = [x for sublist in list(case_test_df.VIOLATED_ARTICLES) for x in sublist]
violated_art_test = [x for x in violated_art_test if 'P' not in x]
violated_art_test_count = pd.value_counts(violated_art_test)
violated_art_test_int = [int(x) for x in violated_art_test]

#%% Plot count of broken articles

plt.plot(violated_art_train_count, label = 'train')
plt.plot(violated_art_dev_count, label = 'dev')
plt.plot(violated_art_test_count, label = 'test')
plt.legend()
plt.show()

#%%

_ = plt.hist(violated_art_train_int, bins = 66, label = 'train', alpha = .5)
_ = plt.hist(violated_art_dev_int, bins = 66, label = 'dev', alpha = .5)
_ = plt.hist(violated_art_test_int, bins = 66, label = 'test', alpha = .5)
plt.legend()
plt.show()

#%% Build heatmap plot

train = []
dev = []
test = []

for x in range(1,67):
    if str(x) in violated_art_train_count.index:
        train.append(violated_art_train_count[str(x)])
    else:
        train.append(0)
    
    if str(x) in violated_art_dev_count.index:
        dev.append(violated_art_dev_count[str(x)])
    else:
        dev.append(0)
    
    if str(x) in violated_art_test_count.index:
        test.append(violated_art_test_count[str(x)])
    else:
        test.append(0)

train_perc = np.array([x / sum(train) for x in train])
dev_perc = np.array([x / sum(dev) for x in dev])
test_perc = np.array([x / sum(test) for x in test])
perc = np.stack((train_perc, dev_perc, test_perc), axis = 0)

#%% Plot percentage of broken articles

_ = plt.plot(train_perc, label = 'train', color = 'red', alpha = .5)
_ = plt.plot(dev_perc, label = 'dev', alpha = .5)
_ = plt.plot(test_perc, label = 'test', alpha = .5)
plt.xlabel('article #')
plt.ylabel('freq')
plt.legend()
plt.grid()
plt.xticks([x for x in range(0,66,3)])
plt.show()

#%% Plot percentage of broken articles (zoom)

_ = plt.plot(train_perc, label = 'train', color = 'red', alpha = .5)
_ = plt.plot(dev_perc, label = 'dev', alpha = .5)
_ = plt.plot(test_perc, label = 'test', alpha = .5)
plt.rcParams["figure.figsize"] = [16,9]
plt.xlabel('article #')
plt.ylabel('freq')
plt.legend()
plt.grid()
plt.xlim([0,20])
plt.xticks([x for x in range(0,21)])
plt.show()

#%% Plot heatmap

#plt.imshow(perc, cmap='hot', interpolation='nearest')   
sns.heatmap(perc)
ax=plt.gca()
ax.set_xlim(0, 60)

#%% EDA ECHR law

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



#-------------------------------------------------------------------
# Extract cases with violated arts
vio_arts = case_train_df.VIOLATED_ARTICLES
case_viol_art_train_df = case_train_df[[x != [] for x in vio_arts]]
vio_par = case_viol_art_train_df.VIOLATED_PARAGRAPHS
case_viol_par_train_df = case_viol_art_train_df[[x != [] for x in vio_par]]


