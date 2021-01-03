
import random
import sklearn
from sklearn import metrics

#%% Binary classification
a = [0] * int(10000 * 0.99)
b = [1] * int(10000 * 0.01)

gold = a + b

pred = []

for i in range(0, len(gold)):
    rand = random.randint(0,1)
    pred.append(rand)
    
#sklearn.metrics.f1_score(gold, pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')

f1 = metrics.f1_score(gold, pred)
precision = metrics.precision_score(gold, pred)
recall = metrics.recall_score(gold, pred)

print(f'precision = {precision * 100:.2f} %')
print(f'recall = {recall * 100:.2f} %')
print(f'f1 = {f1 * 100:.2f} %')

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