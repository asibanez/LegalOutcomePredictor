import torch
import torch.nn.functional as F


logits = torch.Tensor([0.8, 0.1, 0])

a, b, c = 0, 0 ,0

for idx in range (0,1000):
    output = F.gumbel_softmax(logits, tau=1, hard=1)
    a += output[0]
    b += output[1]
    c += output[2]
                
print(a,b,c)
    