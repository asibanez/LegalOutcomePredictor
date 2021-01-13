#%% Imports
import os
import pickle
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve

#%% Function definition

def compute_metrics(Y_ground_truth, Y_predicted_binary, Y_predicted_score):
    tn, fp, fn, tp = confusion_matrix(Y_ground_truth, Y_predicted_binary).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    auc = roc_auc_score(Y_ground_truth, Y_predicted_score)
    
    return precision, recall, f1, auc

#%% Path definitions

base_path = os.getcwd()
input_path = os.path.join(base_path, '01_data', '02_results','00_no_attention', 'results.pkl')
#input_path = os.path.join(base_path, '01_data', '02_results','01_attention', 'results.pkl')

#%% Read data

with open(input_path, 'rb') as fr:
    results = pickle.load(fr)
    
Y_predicted_score = results['Y_test_prediction_scores']
Y_ground_truth = results['Y_test_ground_truth']

#%% Compute class balances:
num_negative = Y_ground_truth.count(0)
num_positive = Y_ground_truth.count(1)
num_total = num_negative + num_positive

share_negative = num_negative / num_total
share_positive = num_positive / num_total

print(f'% negative = {share_negative*100:.2f}')
print(f'% positive = {share_positive*100:.2f}')

#%% Generate random results for unbalanced dataset

random_pred_score = []

for i in range(0, len(Y_predicted_score)):
    random_pred_score.append(random.random())
    
#sklearn.metrics.f1_score(gold, pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
random_pred_binary = [1 if x >= 0.5 else 0 for x in random_pred_score]

#%% Plot ROC curve
fpr_model, tpr_model, threshold_roc_model = roc_curve(Y_ground_truth, Y_predicted_score)
fpr_rand, tpr_rand, threshold_roc_rand = roc_curve(Y_ground_truth, random_pred_score)
plt.plot(fpr_model, tpr_model, linestyle='--', label='Model')
plt.plot(fpr_rand, tpr_rand, linestyle=':', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.grid()
plt.show()    

#%% Plot Precision - recall curve
precision_model, recall_model, threshold_model = precision_recall_curve(Y_ground_truth, Y_predicted_score)
precision_rand, recall_rand, threshold_rand = precision_recall_curve(Y_ground_truth, random_pred_score)
plt.plot(recall_model, precision_model, linestyle='--', label='Model')
plt.plot(recall_rand, precision_rand, linestyle=':', label='Random')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc = 'lower left')
plt.grid()
plt.show()

#%% Compute metrics

# Model
threshold = 0.5
Y_predicted_binary = [1 if x >= threshold else 0 for x in Y_predicted_score]
precision_model, recall_model, f1_model, auc_model = compute_metrics(Y_ground_truth,
                                                                     Y_predicted_binary,
                                                                     Y_predicted_score)

print(f'Precision model: {precision_model:.3f}')
print(f'Recall model: {recall_model:.3f}')
print(f'F1 model: {f1_model:.3f}\n')
print(f'AUC model: {auc_model:.3f}\n')

# Random classifier
#f1 = metrics.f1_score(gold, pred_binary)
#precision = metrics.precision_score(gold, pred_binary)
#recall = metrics.recall_score(gold, pred_binary)

precision_rand, recall_rand, f1_rand, auc_rand = compute_metrics(Y_ground_truth,
                                                                 random_pred_binary,
                                                                 random_pred_score)
print(f'Precision rand: {precision_rand:.3f}')
print(f'Recall rand: {recall_rand:.3f}')
print(f'F1 rand: {f1_rand:.3f}\n')
print(f'AUC rand: {auc_rand:.3f}\n')


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
   
