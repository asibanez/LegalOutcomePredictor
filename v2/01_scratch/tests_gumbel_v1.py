import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

#%%
logits = torch.Tensor([0.8, 0.1, 0])

a, b, c = 0, 0 ,0

for idx in range (0,1000):
    output = F.gumbel_softmax(logits, tau=1, hard=1)
    a += output[0]
    b += output[1]
    c += output[2]
    print(output)               
print(a,b,c)


#%%

logits = torch.Tensor([-500, 500])

a, b, = 0, 0

for idx in range (0,10000):
    output = F.gumbel_softmax(logits, tau=1, hard=1)
    a += output[0]
    b += output[1]
print(a,b)




