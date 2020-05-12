import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from PIL import Image
#
# image = Image.open('download.jpeg')
#
# normalize = transforms.Normalize(mean=[0.456],
#                                  std=[0.224])
#
# train_transform = transforms.Compose([
#     # transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     # transforms.Grayscale(num_output_channels=1),
#     transforms.ToTensor(),
#     normalize, ])
#
# input = train_transform(image)
# input = input.unsqueeze(0)
#
# focal_input = torch.nn.functional.grid_sample(input, (torch.Tensor(1, 100, 100, 2)))
# print(input.shape)
# print(focal_input.shape)
# plt.imshow((input[0].permute(1, 2, 0)))
# plt.show()
# plt.imshow(focal_input[0].permute(1, 2, 0))
# plt.show()


input = torch.arange(4*4).view(1, 1, 4, 4).float()
print(input)


# Create grid to upsample input
d = torch.linspace(1, 1, 2)
meshx, meshy = torch.meshgrid((d, d))
grid = torch.stack((meshy, meshx), 2)
grid = grid.unsqueeze(0) # add batch dim
print(grid.shape)
print(grid)
output = torch.nn.functional.grid_sample(input, grid)
print(output)
