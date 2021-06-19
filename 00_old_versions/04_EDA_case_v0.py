#%% Imports
import os
import json
import glob
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

#%% Path definition
data_folder = 'C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\01_Local\\05_Liberty_Mutual\\Research\\02_legal_outcome_predictor\\00_data\\01_cases\\EN_train\\'
input_path = os.path.join(data_folder, '*.json')

#%% Variable initialization
data_df = pd.DataFrame(columns = ['ITEMID',
                                  'LANGUAGEISOCODE',
                                  'RESPONDENT',
                                  'BRANCH',
                                  'DATE',
                                  'DOCNAME',
                                  'IMPORTANCE',
                                  'CONCLUSION',
                                  'JUDGES',
                                  'TEXT',
                                  'VIOLATED_ARTICLES',
                                  'VIOLATED_PARAGRAPHS',
                                  'VIOLATED_BULLETPOINTS',
                                  'NON_VIOLATED_ARTICLES',
                                  'NON_VIOLATED_PARAGRAPHS',
                                  'NON_VIOLATED_BULLETPOINTS'])
               
#%% Data loading and dataframe construction
for file in tqdm.tqdm(glob.glob(input_path)[0:500]):
    with open(file) as fr:
        data = json.load(fr)
        data_df = data_df.append(data, ignore_index = True)

#%% Compute number of sentences in text
case_texts = data_df.TEXT.to_list()
num_cases = len(case_texts)
num_sent_per_case = [len(x) for x in case_texts]     
case_whole_texts = [' '.join(x) for x in case_texts]
case_tokens = [x.split(' ') for x in case_whole_texts]
num_tokens_per_case = [len(x) for x in case_tokens]

#%% Plot number of sentences per case
fig = plt.figure()
plt.xlabel('# sentences per case')
plt.ylabel('freq')
plt.hist(num_sent_per_case)
fig.show()

#%% Plot number of tokens per case
fig = plt.figure()
plt.xlabel('# tokens per case')
plt.ylabel('freq')
plt.hist(num_tokens_per_case)
fig.show()

#%% Compute value counts for violated articles
aux = data_df['VIOLATED_ARTICLES'].str.join(sep=',')
print(pd.value_counts(aux))

#%%
aux_1 = data_df['VIOLATED_ARTICLES'].str.join(sep=',')
aux_2 = data_df['VIOLATED_PARAGRAPHS'].str.join(sep=',')

data_df[(aux_1 == '4') & (aux_2 == '')]
data_df[(aux_1 == '2')]

#pprint.pprint(list((data_df[(aux_1 == '2')]['CONCLUSION'])))
