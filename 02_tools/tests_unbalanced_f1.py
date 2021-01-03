
import random
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve

#%% Global initialization
random.seed(1234)
negative_share = 0.99
#negative_share = 0.5

#%% Binary classification
a = [0] * int(10000 * negative_share)
b = [1] * int(10000 * (1 - negative_share))

gold = a + b
random.shuffle(gold)

pred = []

for i in range(0, len(gold)):
    rand = random.randint(0,100)
    pred.append(rand/100)
    
#sklearn.metrics.f1_score(gold, pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')

pred_binary = [1 if x >= 0.5 else 0 for x in pred]

f1 = metrics.f1_score(gold, pred_binary)
precision = metrics.precision_score(gold, pred_binary)
recall = metrics.recall_score(gold, pred_binary)

print(f'precision = {precision * 100:.2f} %')
print(f'recall = {recall * 100:.2f} %')
print(f'f1 = {f1 * 100:.2f} %')

#%% Plot ROC curve
fpr, tpr, threshold_roc = roc_curve(gold, pred)
plt.plot(fpr, tpr, linestyle='--', label='Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.grid()
plt.show()    

# Plot Precision - recall curve
precision_model, recall_model, threshold = precision_recall_curve(gold, pred)
plt.plot(recall_model, precision_model, linestyle='--', label='Model')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc = 'lower left')
plt.grid()
plt.show()

#%% Multiclass classification
a = [0] * 5000
b = [1] * 5000
c = [2] * 5000
d = [3] * 5000

pred = a + b + c + d

gold = []

for i in range(0, 20000):
    rand = random.randint(0,3)
    gold.append(rand)
    
f1 = metrics.f1_score(gold, pred, average = 'weighted')
print(f1)

#%% Multiclass classification
a = [0] * 100
b = [1] * 100
c = [2] * 14800

pred = a + b + c

gold = []

for i in range(0, 15000):
    rand = random.randint(0,2)
    gold.append(rand)
    
f1 = metrics.f1_score(gold, pred, average = 'weighted')
print(f1)