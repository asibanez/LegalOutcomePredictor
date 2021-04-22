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

#%%

str_data = {'case_id':[], 'article_id':[], 'broken':[]}

for case in case_test_df.iterrows():
    violated_arts = case[1]['VIOLATED_ARTICLES']
    case_id = case['ITEMID'] * 67
    art_id = list(range(1,68))
    outome = [0] * 67
    
    for violated_art in violated_arts:
        art_id = int(violated_art)
        outcome[art_id] True
        
        
        
        art_text = ECHR_dict[art_id]
        str_data['art_id'].append(art_id)
        str_data['art_text'].append(art_text)
        str_data['case_texts'].append(case_text)
    else:
        str_data['art_id'].append(art_id)
        str_data['art_text'].append(art_text)
        str_data['case_texts'].append(case_text)

                
                
                
#%%
 
str_data = pd.DataFrame(columns = ['art_id', 'art_text', 'case_texts'])          

for case in case_test_df.iterrows():
    case_text = case[1]['TEXT']
    violated_arts = case[1]['VIOLATED_ARTICLES']
    if violated_arts != []:
        for violated_art in violated_arts:
            art_id = violated_art
            art_text = ECHR_dict[int(art_id)]
            if art_id in str_data['art_id']:
                article
                # Add case text to existing violated article
                #str_data[art_id]['case_texts'].append(case_text)
                #str_data
                continue
            else:
                # Add violated article with corresponding case text
                str_data = str_data.append({'art_id':art_id,
                                            'art_text':art_text,
                                            'case_text':case_text}, ignore_index=True)

#%%
a=[]

for item in a:
    print(item)
#%%


