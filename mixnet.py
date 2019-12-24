from torch import nn
from torch import cat
from collections import OrderedDict

class mixBlock(nn.Module):
    """For layers v1 through IT"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride=stride) # padding?
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, input):
        x = self.conv(input)
        x = self.nonlin(x)
        x = self.pool(x)

        return x


class Retina_LGN(nn.Module):
    """Retina and LGN layers in one block because of special structure"""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.midget_conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=3, stride=1)
        self.midget_pool = nn.AdaptiveMaxPool2d(output_size=103, return_indices=False)

        self.parasol_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2)

        self.magno_conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1)
        self.magno_conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)

        self.parvo_conv1 = nn.Conv2d(in_channels=320, out_channels=512, kernel_size=3, stride=1)
        self.parvo_conv2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1)
        self.parvo_dimred1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1)
        self.parvo_conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
        self.parvo_dimred2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        self.parvo_conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
        self.parvo_dimred3 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1)

    def forward(self, input):

        midget_features = self.midget_conv1(input)

        midget_features = self.midget_pool(midget_features)

        parasol_features = self.parasol_conv(input)

        magno_features1 = self.magno_conv1(parasol_features)
        magno_features2 = self.magno_conv2(magno_features1)

        parvo_features1 = self.parvo_conv1(cat([magno_features2, midget_features], dim=1))
        parvo_features2 = self.parvo_conv2(parvo_features1)
        parvo_features2 = self.parvo_dimred1(parvo_features2)
        parvo_features3 = self.parvo_conv3(parvo_features2)
        parvo_features3 = self.parvo_dimred2(parvo_features3)
        parvo_features4 = self.parvo_conv4(parvo_features3)
        parvo_features4 = self.parvo_dimred3(parvo_features4)

        return parvo_features4


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


def mixnetV1():
    model = nn.Sequential(
        OrderedDict([
            ('RetinaLGN', Retina_LGN(3, 128)),
            ('V1', mixBlock(128, 196)),
            ('V2', mixBlock(196, 256)),
            ('V3', mixBlock(256, 324)),
            ('V4', mixBlock(324, 512)),
            ('IT', mixBlock(512, 1024)),
            ('decoder', nn.Sequential(OrderedDict([
                ('avgpool', nn.AdaptiveAvgPool2d(1)),
                ('flatten', Flatten()),
                ('linear', nn.Linear(1024, 1000))
            ])))
        ]))
    return model




