import torch
from sklearn.metrics import classification_report

y_pred = [[1, 1, 0, 1],[1, 1, 0, 1]]
y_true = [[1, 1, 0, 1],[1, 1, 0, 1]]

print(classification_report(y_true, y_pred, labels=[1, 2, 3, 4]))
print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3]))
print(classification_report(y_true, y_pred))


y_pred = ['a', 'c', 'a', 'b']
y_true = ['a', 'c', 'a', 'b']

print(classification_report(y_true, y_pred, labels=['a', 'b', 'c']))
print(classification_report(y_true, y_pred))


y_pred = torch.Tensor([[1, 1, 1, 1],[1, 1, 1, 1]])
y_true = torch.Tensor([[1, 1, 1, 1],[1, 1, 0, 1]])
print(classification_report(y_true, y_pred))

kuku = classification_report(y_true, y_pred)

