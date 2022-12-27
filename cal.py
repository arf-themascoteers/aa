from torchvision.datasets import Caltech256
from torchvision import datasets, transforms
import torch

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.CenterCrop(178),
    # transforms.Resize(64),
])
cal_data = Caltech256(root='./data', download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=cal_data,
                                          batch_size=64,
                                          shuffle=True)

print(cal_data)