import torch

tensor_a = torch.tensor([[[1,2,3],[1,2,3],[1,2,3]],[[4,5,6],[4,5,6],[4,5,6]]])
tensor_b = torch.tensor([[7,8,9],[7,8,9],[7,8,9]])

print(tensor_a)
print(tensor_b)

tensor_a = tensor_a.view([1,-1,3]).squeeze(0)

print(tensor_a)
print(tensor_b)

concatenated = torch.cat([tensor_a, tensor_b], dim = 0)

print(concatenated)

#%%

tensor_a = torch.tensor([[[1],[1],[1]],[[2],[2],[2]]])
tensor_b = torch.tensor([[3],[3],[3]])

print(tensor_a)
print(tensor_b)

tensor_a = tensor_a.view([1,-1,1]).squeeze(0)

print(tensor_a)
print(tensor_b)

concatenated = torch.cat([tensor_a, tensor_b], dim = 0)

print(concatenated)

tensor_b.view([3])
