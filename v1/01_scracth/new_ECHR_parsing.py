import os
import nltk
import pandas as pd

path = 'C://Users//siban//Dropbox//CSAIL//Projects//13_Legal_Outcome_Predictor//00_data//01_preprocessed//ECHR_paragraphs.csv'

data = pd.read_csv(path)

text=list(data.Text)

for x in text: print(x+'\n')

#%%
test = text[166]
print(f'{test}\n')
segmented = nltk.tokenize.sent_tokenize(test)
for idx, x in enumerate(segmented): print(f'{idx}\t{x}')
#%%