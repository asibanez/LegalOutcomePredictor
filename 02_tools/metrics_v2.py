# v1  -> Bug corrected in call to compute metrics
# v2  -> Plots learning curves

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
input_path = os.path.join(base_path, '01_data', '02_runs','39_art_3-5-6_50p_art_dim_100_20_ep', 'results.pkl')

#%% Global initialization

random.seed(1234)
threshold = 0.5

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

#%% Generate random results for unbalanced dataset

random_pred_score = []

for i in range(0, len(Y_predicted_score)):
    random_pred_score.append(random.random())
    
random_pred_binary = [1 if x >= 0.5 else 0 for x in random_pred_score]

#%% Compute metrics

# Model
Y_predicted_binary = [1 if x >= threshold else 0 for x in Y_predicted_score]
precision_model, recall_model, f1_model, auc_model = compute_metrics(Y_ground_truth,
                                                                     Y_predicted_binary,
                                                                     Y_predicted_score)

# Random classifier
precision_rand, recall_rand, f1_rand, auc_rand = compute_metrics(Y_ground_truth,
                                                                 random_pred_binary,
                                                                 random_pred_score)

#%% Compute f1 scores based on threshold

threshold_list = []
precision_list = []
recall_list = []
f1_list = []

for threshold in range(1, 99, 1):
    threshold = threshold / 100
    Y_predicted_binary = [1 if x >= threshold else 0 for x in Y_predicted_score]
    precision, recall, f1, _, = compute_metrics(Y_ground_truth,
                                                Y_predicted_binary,
                                                Y_predicted_score)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    threshold_list.append(threshold)

#%% Plot learning curves
plt.plot(results['training_loss'], label = 'train')
plt.plot(results['validation_loss'], label = 'validation')
plt.xlabel('Epochs')
plt.legend(loc = 'lower left')
plt.show()

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
precision_model_g, recall_model_g, threshold_model_g = precision_recall_curve(Y_ground_truth, Y_predicted_score)
precision_rand_g, recall_rand_g, threshold_rand_g = precision_recall_curve(Y_ground_truth, random_pred_score)
plt.plot(recall_model_g, precision_model_g, linestyle='--', label='Model')
plt.plot(recall_rand_g, precision_rand_g, linestyle=':', label='Random')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc = 'upper right')
plt.grid()
plt.show()
    
#%% Plot metrics vs threshold

plt.plot(threshold_list, f1_list, label = 'f1')
plt.plot(threshold_list, precision_list, label = 'precision')
plt.plot(threshold_list, recall_list, label = 'recall')
plt.xlabel('Threshold')
plt.legend(loc = 'lower left')
plt.show()

#%% Print results

print(f'\n% negative = {share_negative*100:.2f}')
print(f'% positive = {share_positive*100:.2f}\n')

print(f'Precision model: {precision_model:.3f}')
print(f'Recall model: {recall_model:.3f}')
print(f'F1 model: {f1_model:.3f}')
print(f'Max F1 model: {max(f1_list):.3f}')
print(f'AUC model: {auc_model:.3f}\n')

print(f'Precision rand: {precision_rand:.3f}')
print(f'Recall rand: {recall_rand:.3f}')
print(f'F1 rand: {f1_rand:.3f}')
print(f'AUC rand: {auc_rand:.3f}\n')
