import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CoAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CoAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.Wq = nn.Linear(input_dim, hidden_dim)
        self.Wk = nn.Linear(input_dim, hidden_dim)
        self.Wv = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x,y):
        q = self.Wq(x)
        k = self.Wk(y)
        v = self.Wv(x)
        score = self.softmax(torch.bmm(q, k.permute(0, 2, 1)) / math.sqrt(self.input_dim))  # QK/k
        weighted_v = torch.bmm(score, v)

        return weighted_v
class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, insnorm, insnorm_threshold):
        insnorm1, insnorm2 = insnorm[0].weight.abs(), insnorm[1].weight.abs()
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[:, insnorm1 >= insnorm_threshold] = x[0][:, insnorm1 >= insnorm_threshold]
        x1[:, insnorm1 < insnorm_threshold] = x[1][:, insnorm1 < insnorm_threshold]
        x2[:, insnorm2 >= insnorm_threshold] = x[1][:, insnorm2 >= insnorm_threshold]
        x2[:, insnorm2 < insnorm_threshold] = x[0][:, insnorm2 < insnorm_threshold]
        return [x1, x2]

class InstanceNorm2dParallel(nn.Module):
    def __init__(self, num_features):
        super(InstanceNorm2dParallel, self).__init__()
        for i in range(2):
            setattr(self, 'insnorm_' + str(i), nn.InstanceNorm2d(num_features, affine=True, track_running_stats=True))

    def forward(self, x_parallel):
        return [getattr(self, 'insnorm_' + str(i))(x) for i, x in enumerate(x_parallel)]

if __name__ == '__main__':
    model = CoAttention(512,256)
    x = torch.randn((16,512,512))
    y = torch.randn((16, 512, 512))
    z = model(x,y)

    # parser.add_argument('-t', '--insnorm-threshold', type=float, default=1e-2,
    #                     help='threshold for slimming BNs')
    insnorm_conv = InstanceNorm2dParallel(256)
    exchange = Exchange()
    insnorm_threshold = 0.01
    insnorm_list = []
    for module in insnorm_conv.modules():
        if isinstance(module, nn.InstanceNorm2d):
            insnorm_list.append(module)
    out = exchange(z, insnorm_list, insnorm_threshold)
    print(out)
