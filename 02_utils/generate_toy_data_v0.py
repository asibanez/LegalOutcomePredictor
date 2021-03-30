# Imports
import os
import datetime
import pandas as pd

# Path definitions
input_folder = '/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/01_preprocessed/01_article_split/art_03_05_06_13_50p_par_att'
output_folder = '/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/01_preprocessed/01_article_split/art_03_05_06_13_50p_par_att/TOY'

path_train_in = os.path.join(input_folder, 'model_train.pkl')
path_dev_in = os.path.join(input_folder, 'model_dev.pkl')
path_test_in = os.path.join(input_folder,'model_test.pkl')

path_train_out = os.path.join(output_folder, 'model_train.pkl')
path_dev_out = os.path.join(output_folder, 'model_dev.pkl')
path_test_out = os.path.join(output_folder,'model_test.pkl')

# Create output folder if not existing
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
    print("Created folder : ", output_folder)

# Global initialization
size_train = 3000
size_dev = 1000
size_test = 1000

# Data load
print(datetime.datetime.now(), 'Loading train data')
train_in = pd.read_pickle(path_train_in)
print(datetime.datetime.now(), 'Loading dev data')
dev_in = pd.read_pickle(path_dev_in)
print(datetime.datetime.now(), 'Loading holdout data')
test_in = pd.read_pickle(path_test_in)
print(datetime.datetime.now(), 'Done')

# Data slicing
train_out = train_in[0:size_train]
dev_out = dev_in[0:size_dev]
test_out = test_in[0:size_test]

print(f'Train size before = {len(train_in):,} / after = {len(train_out):,}')
print(f'Dev size before = {len(dev_in):,} / after = {len(dev_out):,}')
print(f'Test size before = {len(test_in):,} / after = {len(test_out):,}')

# Save outputs
train_out.to_pickle(path_train_out)
dev_out.to_pickle(path_dev_out)
test_out.to_pickle(path_test_out)
