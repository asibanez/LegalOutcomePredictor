import torch
import numpy as np

mask = torch.Tensor([[1,1,0,0,1],[0,0,0,0,0],[1,1,1,1,1],[1,1,1,0,0]])

mask = mask.numpy()

rationales = []
for idx in range(0,mask.shape[0]):
    mask_single = mask[idx, :]
    output_aux = np.where(mask_single == 1)[0].tolist()
    rationales.append(output_aux)
    
    
