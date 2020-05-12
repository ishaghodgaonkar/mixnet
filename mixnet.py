from torch import nn
from collections import OrderedDict
import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torchvision.transforms import transforms

class mixBlock(nn.Module):
    """For layers V1 through IT"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size = kernel_size, stride=stride, padding=1)
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.dimred = nn.Conv2d(in_channels*2, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, input):
        x_conv = self.conv(input)
        x_nonlin = self.nonlin(x_conv)
        x = self.pool(x_nonlin)
        input_pooled = self.pool(input)
        x = torch.cat((x, input_pooled), 1)
        #x = self.dimred(x)
        return x


class Retina(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        """
        Horizontal cells activate the bipolar cell layer using lateral inhibition. This effectively increases 
        contrast at edges. To model this, the kernel learned in the bipolar convolutional layer should be some type of 
        edge detection or sharpen kernel. 
        """
        
        """
        rods outnumber cones in the retina, but cones are more concentrated in the fovea. cones are wider than rods.
        """
        #weights = weights.view(1, 1, 3, 3).repeat(16, 1, 1, 1)
        self.rods = nn.Conv2d(1, 16, kernel_size=3, stride=1, bias=False)
        self.cones = nn.Conv2d(in_channels, 16, kernel_size=7, stride=1, bias=False)
        #self.bipolar_conv.weight = nn.Parameter(weights, requires_grad=True)

        """
        Ganglion cells take input from the bipolar cell layer. These are in the form of circular receptive fields, 
        either the middle being inhibitory and surrounding excitatory, or the opposite. These include midget cells 
        (large receptive field, bigger detail detection) and parvocellular cells (for fine detail and color processing).
        Since we are only dealing with grayscale images, try to model migdet cells with larger receptive field size.
        
        """
        """
        bipolar cells consolidate information from rods and cones before passing to ganglion cells
        """
        self.bipolar = nn.Conv2d(32, out_channels, kernel_size=5, stride=2)
        # self.ganglion = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        """
        Similar receptive fields as ganglion cells (circular, activated in the center or in periphery). 
        """

        #self.LGN_conv = nn.Conv2d(32, out_channels, kernel_size=3, stride=1)

    def forward(self, input):
        trans = transforms.ToPILImage()
        tensor_trans = transforms.ToTensor()
        grayscale_trans = transforms.Grayscale()

        # get middle ~1/2 of image, grayscale whole image
        final_focal_input_tensor = torch.Tensor()
        final_whole_input_tensor = torch.Tensor()


        for i in range(input.shape[0]):
            input = input.cpu()
            PIL_image = trans(input[0])
            focal_input = PIL_image.crop((37, 37, 187, 187))
            focal_input = tensor_trans(focal_input).unsqueeze(0)
            final_focal_input_tensor = torch.cat((final_focal_input_tensor, focal_input), 0)

            grayscaled_input = grayscale_trans(PIL_image)
            grayscaled_input = tensor_trans(grayscaled_input).unsqueeze(0)
            final_whole_input_tensor = torch.cat((final_whole_input_tensor, grayscaled_input), 0)

        # plt.imshow((input[0].permute(1, 2, 0)))
        # plt.show()
        # plt.imshow(final_focal_input_tensor[0].permute(1, 2, 0))
        # plt.show()

        # apply rods on whole image, grayscaled
        x_rods = self.rods(final_whole_input_tensor)

        # apply cones on center of image
        x_cones = self.cones(final_focal_input_tensor)
        print(x_rods.shape)
        print(x_cones.shape)

        # upsample cones feature map
        upsample = torch.nn.Upsample(size=222)
        x_cones = upsample(x_cones)
        x = torch.cat((x_rods, x_cones), 1)

        # @todo: more kernel sizes?

        # bipolar layer works on the output of both of these
        x = self.bipolar(x)

        return x

class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


def mixnetV1():
    model = nn.Sequential(
        OrderedDict([
           ('Retina', Retina(3, 64)),
            ('V1', mixBlock(64, 128)),
            ('V2', mixBlock(128, 256)),
            ('V4', mixBlock(256, 512)),
            ('IT', mixBlock(512, 1024)),
            ('decoder', nn.Sequential(OrderedDict([
                ('avgpool', nn.AdaptiveAvgPool2d(1)),
                ('flatten', Flatten()),
                ('linear2', nn.Linear(1024, 1000))
            ])))
        ]))
    return model




