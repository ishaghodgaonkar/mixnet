from torch import nn
from collections import OrderedDict
import torch

class mixBlock(nn.Module):
    """For layers V1 through IT"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size = kernel_size, stride=stride, padding=1)
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.dimred = nn.Conv2d(in_channels*2, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, input):
        x = self.conv(input)
        x = self.nonlin(x)
        x = self.pool(x)
        x = torch.cat((x, input), 1)
        x = self.dimred(x)
        return x


class Retina(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        """
        Horizontal cells activate the bipolar cell layer using lateral inhibition. This effectively increases 
        contrast at edges. To model this, the kernel learned in the bipolar convolutional layer should be some type of 
        edge detection or sharpen kernel. 
        """
        weights = torch.tensor([[-1., -1., -1.],
                                [-1., 8., -1.],
                                [-1., -1., -1.]])
        weights = weights.view(1, 1, 3, 3).repeat(16, 1, 1, 1)
        self.bipolar_conv = nn.Conv2d(in_channels, 16, kernel_size =3, stride=1, bias=False)
        self.bipolar_conv.weight = nn.Parameter(weights)

        """
        Ganglion cells take input from the bipolar cell layer. These are in the form of circular receptive fields, 
        either the middle being inhibitory and surrounding excitatory, or the opposite. These include midget cells 
        (large receptive field, bigger detail detection) and parvocellular cells (for fine detail and color processing).
        Since we are only dealing with grayscale images, try to model migdet cells with larger receptive field size.
        
        """

        self.ganglion_conv = nn.Conv2d(16, 24, kernel_size=5, stride=1)

        """
        Similar receptive fields as ganglion cells (circular, activated in the center or in periphery). 
        """

        self.LGN_conv = nn.Conv2d(24, out_channels, kernel_size=3, stride=1)

    def forward(self, input):

        x = self.bipolar_conv(input)
        x = self.ganglion_conv(x)
        x = self.LGN_conv(x)

        return x

class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


def mixnetV1():
    model = nn.Sequential(
        OrderedDict([
            ('Retina', Retina(1, 32)),
            ('V1', mixBlock(32, 48)),
            ('V2', mixBlock(48, 64)),
            ('V3', mixBlock(64, 96)),
            ('V4', mixBlock(96, 128)),
            ('IT', mixBlock(128, 192)),
            ('decoder', nn.Sequential(OrderedDict([
                ('avgpool', nn.AdaptiveAvgPool2d(1)),
                ('flatten', Flatten()),
                ('linear1', nn.Linear(192, 4096)),
                ('linear2', nn.Linear(4096, 1000))
            ])))
        ]))
    return model




