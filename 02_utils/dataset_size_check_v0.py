import os
import pickle
import pandas as pd

path = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/01_data/02_runs/12_art_6_300_pass'
path_train = os.path.join(path, 'model_train.pkl')
path_dev = os.path.join(path, 'model_dev.pkl')
path_test = os.path.join(path, 'model_test.pkl')

#%%
with open(path_train, 'rb') as fr:
    data_train = pickle.load(fr)
    print('Train size =', data_train.shape)

with open(path_dev, 'rb') as fr:
    data_dev = pickle.load(fr)
    print('Dev size =', data_dev.shape)

with open(path_test, 'rb') as fr:
    data_test = pickle.load(fr)
    print('Test size =', data_test.shape)
