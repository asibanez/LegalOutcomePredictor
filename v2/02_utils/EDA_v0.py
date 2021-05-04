#%% Imports
from datasets import load_dataset
import matplotlib.pyplot as plt

#%% Data loading
train_set = load_dataset('ecthr_cases', split = 'train')
val_set = load_dataset('ecthr_cases', split = 'validation')
test_set = load_dataset('ecthr_cases', split = 'test')

#%% EDA Facts
train_facts = train_set['facts']
val_facts = val_set['facts']
test_facts = test_set['facts']

#%% Compute number of paragraphs
lens_train = [len(x) for x in train_facts]
lens_val = [len(x) for x in val_facts]
lens_test = [len(x) for x in test_facts]
lens_all = lens_train + lens_val + lens_test

#%% Compute maximum number of paragraphs
max_len_train = max(lens_train)
max_len_val = max(lens_val)
max_len_test = max(lens_test)
max_len_all = max(lens_all)

#%% Plot histogram # paragraphs and print results
_ = plt.hist(lens_all, bins = 50, range = [0,200])

print(f'Max number of paragraphs in train = {max_len_train}')
print(f'Max number of paragraphs in val = {max_len_val}')
print(f'Max number of paragraphs in test = {max_len_test}')
print(f'Max number of paragraphs = {max_len_all}')

#%% EDA labels
train_labels = train_set['labels']
val_labels = val_set['labels']
test_labels = test_set['labels']

#%%
train_labels_all = [x for sublist in train_labels for x in sublist]
train_labels_unique = sorted(list(set(train_labels_all)))

val_labels_all = [x for sublist in val_labels for x in sublist]
val_labels_unique = sorted(list(set(val_labels_all)))

test_labels_all = [x for sublist in test_labels for x in sublist]
test_labels_unique = sorted(list(set(test_labels_all)))

all_labels = train_labels_all + val_labels_all + test_labels_all
all_labels_unique = sorted(list(set(all_labels)))

print(f'Number of labels = {len(all_labels_unique)}')

_ = plt.hist(all_labels)

#%% Scracth
_ = plt.hist(lens_all, bins = 50)

#%%
num_par = [x > 200 for x in lens_all]
print(sum(num_par))
#%%


