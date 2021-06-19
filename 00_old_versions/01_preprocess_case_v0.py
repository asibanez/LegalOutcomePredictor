#%% Imports
import os
import json
import glob
import tqdm
import pickle
import pandas as pd

#%% Path definition

base_folder = os.getcwd()
input_path_base = os.path.join(base_folder, '01_data', '00_original', '00_cases')
output_path_base = os.path.join(base_folder, '01_data', '01_preprocessed')
train_dev_test_folders = ['EN_train', 'EN_dev', 'EN_test']

#%% Variable initialization

for folder in tqdm.tqdm(train_dev_test_folders):          
    input_path = os.path.join(input_path_base, folder, '*.json' )
    output_path = os.path.join(output_path_base, 'case_' + folder)
    data_df = pd.DataFrame(columns = ['ITEMID',
                                      'LANGUAGEISOCODE',
                                      'RESPONDENT',
                                      'BRANCH',
                                      'DATE',
                                      'DOCNAME',
                                      'IMPORTANCE',
                                      'CONCLUSION',
                                      'JUDGES',
                                      'TEXT',
                                      'VIOLATED_ARTICLES',
                                      'VIOLATED_PARAGRAPHS',
                                      'VIOLATED_BULLETPOINTS',
                                      'NON_VIOLATED_ARTICLES',
                                      'NON_VIOLATED_PARAGRAPHS',
                                      'NON_VIOLATED_BULLETPOINTS'])
                   
    # Data loading and dataframe construction
    ### Slice files for testing
    file_list = glob.glob(input_path)#[0:1000]
    ###
    for file in tqdm.tqdm(file_list):
        with open(file) as fr:
            data = json.load(fr)
            data_df = data_df.append(data, ignore_index = True)
    
    # Serialize and save dataframe
    with open(output_path, 'wb') as fw:
        pickle.dump(data_df, fw)
