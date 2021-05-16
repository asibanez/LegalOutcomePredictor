# v1  -> Bug corrected in call to compute metrics
# v2  -> Plots learning curves
# v3  -> Reads result in jston format. Input path updated
# v4  -> Converts to multilabel classification task and computes metrics

#%% Imports
import os
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import warnings
warnings.filterwarnings("ignore")

#%% Function definition
def compute_metrics(Y_ground_truth, Y_pred_binary, Y_pred_score):
    tn, fp, fn, tp = confusion_matrix(Y_ground_truth, Y_pred_binary).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    auc = roc_auc_score(Y_ground_truth, Y_pred_score)
    
    return precision, recall, f1, auc

#%% Path definitions
base_path_run = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/v2/02_runs/18_BERT_TRANSF_v2_FIX_50par_20ep_rationale_v1_TEST_DELETE'
path_raw = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/v2/01_preprocessed/01_50pars_256_tok/01_full'

#%% Global initialization
random.seed(1234)
threshold = 0.5
tgt_labels = ['2',
              '3',
              '4',
              '5',
              '6',
              '7',
              '8',
              '9',
              '10',
              '11',
              '12',
              '13',
              '14',
              '15',
              '17',
              '18',
              '34',
              '38',
              '39',
              '46',              
              'P1-1',
              'P1-2',
              'P1-3',
              'P3-1',
              'P4-2',
              'P4-4',
              'P6-3',
              'P7-1',
              'P7-2',
              'P7-3',
              'P7-4',
              'P7-5',
              'P12-1']

#%% Define data path
results_path = os.path.join(base_path_run, 'full_results_model_dev.json')

#%% Load datasets
with open(results_path) as fr:
    results = json.load(fr)

ground_truth = load_dataset('ecthr_cases', split = 'validation')
ground_truth = load_dataset('ecthr_cases', split = 'test')

#%% Extract info
Y_pred_score = results['Y_test_prediction_scores']
Y_ground_truth = results['Y_test_ground_truth']

gt_facts = ground_truth['facts']
gt_labels = ground_truth['labels']
gt_rationales = ground_truth['gold_rationales']

#%% Plot learning curves
plt.plot(results['training_loss'], label = 'train')
plt.plot(results['validation_loss'], label = 'validation')
plt.xlabel('Epochs')
plt.legend(loc = 'upper right')
#plt.ylim(0.05, 0.75)
plt.grid()
plt.show()

#%% Compute classification results
Y_pred_binary = [[1 if x >= threshold else 0 for x in sublist] for sublist in Y_pred_score]
Y_ground_truth = [[int(x) for x in sublist] for sublist in Y_ground_truth]
print(classification_report(Y_ground_truth, Y_pred_binary,
      target_names = tgt_labels))

#%% Extract rationales
Y_rationales = results['Y_test_prediction_rationale']
#Y_gold_rationales = 

#%% Compute initial rationale stats
avg_num_rationales = sum([sum(x) for x in Y_rationales]) / len(Y_rationales)
print(f'Average number of rationales = {avg_num_rationales}')

#%% Remove rationales for entries with no labels
aux_list = []
for label, rationale in zip(Y_pred_binary, Y_rationales):
    if sum(label) == 0:
        aux_list.append([])
    else:
        aux_list.append(rationale)
    
Y_rationales = aux_list

#%% Compute clean rationale stats
avg_num_rationales = sum([sum(x) for x in Y_rationales]) / len(Y_rationales)
print(f'Average number of rationales after cleaning = {avg_num_rationales}')






"""