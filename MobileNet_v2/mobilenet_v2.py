import torch
import torch.nn as nn
from torch.autograd import Variable


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expansion=6):
        super(InvertedResidual, self).__init__()
        assert isinstance(expansion, int), "expansion in the bottleneck should be an int"
        self.stride = stride
        self.expansion = expansion
        self.res_addition = (stride == 1) and (in_channel == out_channel)
        mid_channel = in_channel * expansion

        self.block = nn.Sequential(
            # depthwise conv
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU6(inplace=True),
            # pointwise conv
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=stride, padding=1, groups=mid_channel),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU6(True),
            # linear bottleneck
            nn.Conv2d(mid_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        out = self.block(x)
        if self.res_addition:
            out += x
        return out


class MobilenetV2(nn.Module):
    '''
    suppose the input is 3*224*224
    '''

    def __init__(self):
        super(MobilenetV2, self).__init__()
        self.network_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.features = [nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU6(True)
        )]

        in_channel = 32
        for t, c, n, s in self.network_setting:
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(in_channel, c, s, t))
                else:
                    self.features.append(InvertedResidual(c, c, 1, t))
                in_channel = c
        self.features.append(nn.Conv2d(in_channel, 1280, 1))
        self.features.append(nn.AvgPool2d(7))
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        x = self.features(x)
        return x


if __name__ == '__main__':
    net = MobilenetV2()
    x = torch.rand(2, 3, 224, 224).float()
    x = Variable(x)
    y = net(x)
    print(y.size())
