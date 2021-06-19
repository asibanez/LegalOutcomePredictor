import os
import pandas as pd

path='C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\01_data\\01_preprocessed\\01_article_split\\art_13_bis'
path=os.path.join(path,'model_train.pkl')

data = pd.read_pickle(path)
case_texts = data.case_texts

print(f'number of passages = {len(case_texts[0])/512}')


