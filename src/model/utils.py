from torch import nn

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()

        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)
