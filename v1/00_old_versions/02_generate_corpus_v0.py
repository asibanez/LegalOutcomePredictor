#%% Imports
import os
import nltk
import tqdm
import codecs
import pickle
import pandas as pd
#from sklearn.feature_extraction.text import CountVectorizer

#%% Path definition

base_folder = os.getcwd()
input_folder = os.path.join(base_folder, '01_data', '01_preprocessed')
output_path = os.path.join(base_folder, '01_data', '01_preprocessed', 'corpus.txt')

train_dev_test_files = ['case_EN_train', 'case_EN_dev', 'case_EN_test']
ECHR_dict_path = os.path.join(input_folder, 'ECHR.dict')
case_train_path = os.path.join(input_folder, 'case_EN_train')
case_dev_path = os.path.join(input_folder, 'case_EN_dev')
case_test_path = os.path.join(input_folder, 'case_EN_test')

#%% Initialize variables

min_token_freq = 0.03

#%% Generate ECHR corpus

with open(ECHR_dict_path, 'rb') as fr:
    ECHR_dict = pickle.load(fr)
    
corpus_ECHR = (' ').join(list(ECHR_dict.values())).lower()

#%% Generate case_train corpus

with open(case_train_path, 'rb') as fr:
    case_df = pickle.load(fr)

case_texts = case_df.TEXT.to_list()
case_text_joined = [x for sublist in case_texts for x in sublist]
corpus_case_train = ' '.join(case_text_joined).lower()

#%% Generate case_dev corpus

with open(case_dev_path, 'rb') as fr:
    case_df = pickle.load(fr)

case_texts = case_df.TEXT.to_list()
case_text_joined = [x for sublist in case_texts for x in sublist]
corpus_case_dev = ' '.join(case_text_joined).lower()

#%% Generate case_test corpus

with open(case_test_path, 'rb') as fr:
    case_df = pickle.load(fr)

case_texts = case_df.TEXT.to_list()
case_text_joined = [x for sublist in case_texts for x in sublist]
corpus_case_test = ' '.join(case_text_joined).lower()

#%% Compute token frequency in corpus

corpus = (' ').join([corpus_ECHR, corpus_case_train, corpus_case_dev, corpus_case_test])
corpus_tok = nltk.word_tokenize(corpus)

"""
#%% V1
token_freq_v1 ={}

for token in tqdm.tqdm(corpus_tok):
    if token not in list(token_freq.keys()):
        token_freq[token] = 1
    else:
        token_freq[token] += 1
"""

#%% V2
token_dist = nltk.FreqDist(corpus_tok)
token_counts = token_dist.most_common()
token_num = len(token_counts)
token_counts_most_freq = token_dist.most_common(int(token_num * (1 - min_token_freq)))

#%% Generate vocabulary

vocab = sorted([x[0] for x in token_counts_most_freq])

#%% Replace low frequency tokens for <UNK>

#corpus_new = [x if x in vocab else '<UNK>' for x in corpus_tok]
corpus_new = []
for token in tqdm.tqdm(corpus_tok):
    if token in vocab:
        corpus_new.append(token)
    else:
        corpus_new.append('<UNK>')

#%% Save corpus to file

with codecs.open(output_path, 'w', 'utf-8') as fw:
    fw.write(corpus_new)
