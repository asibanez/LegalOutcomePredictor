# Computes stats for test results

#%% Imports
import json
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#%% Path definitions
path_results = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/02_runs/batch_02/art_06/11_art6_50p_att_v4_2_50ep_att_TEST/full_results.json'
path_test_set_preprocessed = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/01_preprocessed/case_EN_test_df.pkl'
path_test_set_model = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/01_preprocessed/01_article_split/art_06_50p_par_att/model_test.pkl'

#%% Global initialization
tp, tn, fp, fn = 0, 0, 0, 0

#%% Load data
with open(path_results, 'r') as fr:
    results = json.load(fr)

with open(path_test_set_model, 'rb') as fr:
    test_model_df = pickle.load(fr)

with open(path_test_set_preprocessed, 'rb') as fr:
    test_preproc_df = pickle.load(fr)

#%%
predictions = [int(x) for x in results['Y_test_prediction_binary']]
ground_truth = [int(x) for x in results['Y_test_ground_truth']]
case_id = test_model_df['case_id']

#%% Build main dataframe
data_df = pd.DataFrame({'case_id': case_id,
                        'predictions':predictions,
                        'ground_truth': ground_truth})

#%% Compute number of case paragraphs per entry
num_case_passages = []
for case_id in data_df['case_id']:
    entry = test_preproc_df[test_preproc_df.ITEMID == case_id]
    num_passages = len(entry['TEXT'].item())
    num_case_passages.append(num_passages)

data_df['num_case_passages'] = num_case_passages

#%% Compute number of violated article paragraphs
num_violated_pars = []
for case_id in data_df['case_id']:
    entry = test_model_df[test_model_df.case_id == case_id]
    num_violated_pars_aux = sum(entry['violated_pars'].item())
    num_violated_pars.append(num_violated_pars_aux)

data_df['num_violated_pars'] = num_violated_pars

#%% Slice wrong and correct predictions
data_df_wrong = data_df[data_df.predictions != data_df.ground_truth]
data_df_correct = data_df[data_df.predictions == data_df.ground_truth]

#%% Compute metrics
for _, entry in data_df.iterrows():
    prediction = entry['predictions']
    ground_truth = entry['ground_truth']
    if prediction == 1 and ground_truth == 1: tp += 1
    if prediction == 0 and ground_truth == 0: tn += 1
    if prediction == 1 and ground_truth == 0: fp += 1
    if prediction == 0 and ground_truth == 1: fn += 1

#%%    
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)

#%% Print metrics
print(f'Precision = {precision}')
print(f'Recall = {recall}')
print(f'F1 = {f1}')

#%% Plot stats
_ = plt.hist(data_df_correct.num_case_passages)
_ = plt.hist(data_df_wrong.num_case_passages)
_ = plt.hist(data_df_correct.num_violated_pars)
_ = plt.hist(data_df_wrong.num_violated_pars)


