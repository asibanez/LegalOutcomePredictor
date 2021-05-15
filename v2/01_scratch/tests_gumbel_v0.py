import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

#%%
logits = torch.Tensor([0.8, 0.1, 0])

a, b, c = 0, 0 ,0

for idx in range (0,1000):
    output = F.gumbel_softmax(logits, tau=1, hard=1)
    #print(sum(output))
    #a += output[0]
    #b += output[1]
    #c += output[2]
    print(output)               
#print(a,b,c)

#%%
logits = torch.Tensor([-1000])

sum_output = 0
for idx in range (0,1000):
    output = F.gumbel_softmax(logits, tau=1, hard=1)
    sum_output += 1
    print(output)
print(sum_output)

#%%
logits = torch.Tensor([0.3, 0.2])

sum_a = 0
sum_b = 0
for idx in range (0,1000):
    output = F.gumbel_softmax(logits, tau=1, hard=1)
    sum_a += output[0]
    sum_b += output[1]
    print(output)
print(sum_a, sum_b)

#%%
logits = torch.Tensor([1, 0.1, 0])

logits = logits.repeat(2,1)
logits = logits.unsqueeze(2)

print(f'logits = {logits}')
output = F.gumbel_softmax(logits, tau=100, hard=False)
print(output)
#%%
plt.bar(['1', '2', '3'], output)


#%%
import torch
import torch.nn.functional as F

temperature = 1
input = torch.Tensor([1,2,3])

noise = torch.rand(input.size())
noise.add_(1e-9).log_().neg_()
noise.add_(1e-9).log_().neg_()
x = (input + noise) / temperature
x = F.softmax(x.view(-1,  x.size()[-1]), dim=-1)

#%%

'''
    -z: torch Tensor where each element probablity of element
    being selected
    -args: experiment level config

    returns: A torch variable that is binary mask of z >= .5
'''
z = torch.Tensor([0.2, 0.6, 0.7])
max_z, ind = torch.max(z, dim=-1)
masked = torch.ge(z, max_z.unsqueeze(-1)).float()
del z
return masked

