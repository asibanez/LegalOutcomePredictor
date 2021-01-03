#%% Imports
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve

#%% Function definition

def compute_metrics(Y_ground_truth, Y_predicted_binary):
    tn, fp, fn, tp = confusion_matrix(Y_ground_truth, Y_predicted_binary).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    auc = roc_auc_score(Y_ground_truth, Y_predicted_score)
    
    return precision, recall, f1, auc

#%% Path definitions

base_path = os.getcwd()
input_path = os.path.join(base_path, '01_data', '02_results','01_attention', 'results.pkl')

#%% Read data

with open(input_path, 'rb') as fr:
    results = pickle.load(fr)
    
#%%

Y_predicted_score = results['Y_test_prediction_scores']
Y_ground_truth = results['Y_test_ground_truth']

#%% Compute class balances:
num_negative = Y_ground_truth.count(0)
num_positive = Y_ground_truth.count(1)
print(f'% negative = {num_negative / (num_negative + num_positive)*100:.2f}')
print(f'% positive = {num_positive / (num_negative + num_positive)*100:.2f}')

#%% Plot ROC curve
fpr, tpr, threshold_roc = roc_curve(Y_ground_truth, Y_predicted_score)
plt.plot(fpr, tpr, linestyle='--', label='Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.grid()
plt.show()    

#%% Plot Precision - recall curve
precision_model, recall_model, threshold = precision_recall_curve(Y_ground_truth, Y_predicted_score)
plt.plot(recall_model, precision_model, linestyle='--', label='Model')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc = 'lower left')
plt.grid()
plt.show()

#%% Compute metrics

threshold = 0.5
Y_predicted_binary = [1 if x >= threshold else 0 for x in Y_predicted_score]
precision, recall, f1, auc = compute_metrics(Y_ground_truth, Y_predicted_binary)
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1: {f1:.3f}\n')
print(f'AUC: {auc:.3f}\n')

#%% Compute f1 scores based on threshold

threshold_list = []
precision_list = []
recall_list = []
f1_list = []

for threshold in range(1, 10, 1):
    threshold = threshold / 10
    Y_predicted_binary = [1 if x >= threshold else 0 for x in Y_predicted_score]
    precision, recall, f1, _, = compute_metrics(Y_ground_truth, Y_predicted_binary)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    threshold_list.append(threshold)
    
#%% Plot
plt.plot(threshold_list, f1_list)
plt.plot(threshold_list, precision_list)
plt.plot(threshold_list, recall_list)
   
