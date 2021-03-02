# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# DataClass definition
class LM_dataset(Dataset):
    def __init__(self, data_df):
        self.str_data_tensor = torch.FloatTensor(data_df.drop(columns = ['full_denial']).values)
        self.full_denial_tensor = torch.Tensor(data_df['full_denial'].to_list())
        
    def __len__(self):
        return self.full_denial_tensor.size()[0]
        
    def __getitem__(self, idx):
        X_str = self.str_data_tensor[idx, :]
        Y = self.full_denial_tensor[idx]
        
        return X_str, Y

# Model definition
class LM_model(nn.Module):
    def __init__(self, args, num_str_vars):
        #super(ECHR_model, self).__init__()
        super().__init__()

        self.dropout = args.dropout
        self.num_str_vars = num_str_vars
        self.output_size = 1
      
        # Dropout
        self.drops = nn.Dropout(self.dropout)
        
        # Batch normalizations
        self.bn1_struct = nn.BatchNorm1d(self.num_str_vars)
        self.bn2_struct = nn.BatchNorm1d(int(self.num_str_vars/3))
                
        # Fully connected layers
        self.fc1_struct = nn.Linear(self.num_str_vars, int(self.num_str_vars/3))
        self.fc2_struct = nn.Linear(int(self.num_str_vars/3), self.output_size)     
        
        # Sigmoid
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, X_str):
        # Layer 1
        x_str = self.bn1_struct(X_str)       # batch_size x num_str_vars
        x_str = self.fc1_struct(x_str)       # batch_size x (num_str_vars/3)
        x_str = F.relu(x_str)                # batch_size x (num_str_vars/3)
        x_str = self.drops(x_str)            # batch_size x (num_str_vars/3)
        
        # Layer 2
        x_str = self.bn2_struct(x_str)       # batch_size x (num_str_vars/3)
        x_str = self.fc2_struct(x_str)       # batch size x output_size
        
        # Sigmoid layer output
        x = self.sigmoid(x_str)              # batch size x output_size
        
        return x
