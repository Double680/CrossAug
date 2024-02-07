import torch
import torch.nn as nn
import torch.nn.functional as F


class HouseHolder(nn.Module):
    
    def __init__(self, args, dim, num):
        super(HouseHolder, self).__init__()
        
        # load parameters info
        self.device = args.device
        self.dtype = args.torch_type
        self.dim = dim
        self.num = num

        self.vectors = nn.Parameter(torch.randn(self.num, self.dim, dtype=self.dtype, device=self.device))


    def residual(self, mat1, mat2):
        if mat1.dim() == 2:
            res = torch.mm(mat1, mat2)
        else:
            batch_mat2 = mat2.repeat(mat1.size(0), 1, 1)
            res = torch.bmm(mat1, batch_mat2)

        return res


    def forward(self, X):
        vectors = F.normalize(self.vectors, 2, dim=-1)
        output = X
        for i in range(self.num):
            vec = torch.unsqueeze(vectors[i], 0)
            reduce_matrix = 2 * torch.mm(vec.T, vec)
            output = output - self.residual(output, reduce_matrix)

        return output
