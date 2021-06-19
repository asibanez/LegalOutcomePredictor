# Imports
import os
import argparse
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

def main():
    
    # Argument parsing    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default = None, type = str,
                        required = True, help = 'input folder')
    parser.add_argument('--min_maj_ratio', default = None, type = float,
                        required = True, help = 'Minority to majority class ratio')
    args = parser.parse_args()  

    # Path definition
    input_path = args.input_folder
    
    # Global initialization
    seed = 1234
    
    # Data load
    data = pd.read_pickle(input_path)
    
    # EDA before resampling
    num_neg = pd.value_counts(data.label)[0]
    num_pos = pd.value_counts(data.label)[1]
    print(f'% neg before resampling = {num_neg/(num_pos+num_neg)*100:.2f}')
    print(f'% pos before resampling = {num_pos/(num_pos+num_neg)*100:.2f}')
    
    # Undersampling
    x_train = data[list(data.columns)[:-1]]
    y_train = data[list(data.columns)[-1:]]
    random_undersampler = RandomUnderSampler(sampling_strategy = args.min_maj_ratio,
                                             random_state = seed)
    x_train_res, y_train_res = random_undersampler.fit_resample(x_train, y_train)
    res_data = pd.concat([x_train_res, y_train_res], axis = 1)
    
    # EDA before resampling
    num_neg = pd.value_counts(res_data.label)[0]
    num_pos = pd.value_counts(res_data.label)[1]
    print(f'% neg before resampling = {num_neg/(num_pos+num_neg)*100:.2f}')
    print(f'% pos before resampling = {num_pos/(num_pos+num_neg)*100:.2f}')
    
    # Save resampled data
    file_name, file_ext = os.path.splitext(os.path.basename(input_path))
    output_filename = file_name + '_resampled_' + str(args.min_maj_ratio) + file_ext
    output_folder = os.path.dirname(input_path)
    output_path = os.path.join(output_folder, output_filename)
    pd.to_pickle(res_data, output_path)

#%% Main
if __name__ == "__main__":
    main()
