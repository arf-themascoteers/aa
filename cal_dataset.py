from torchvision.datasets import USPS
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


class CalDataset:
    def __init__(self, is_train=True):
        torch.manual_seed(0)
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.CenterCrop(178),
            transforms.Resize((224,224))
        ])
        cal_data = USPS(root='./data', train=True, download=True, transform=transform)
        size = len(cal_data)
        train_size = int(size*.9)
        test_size = int(size - train_size)

        self.ds, test_ds = torch.utils.data.random_split(cal_data, (train_size, test_size))

        if not is_train:
            self.ds = test_ds


    def get_ds(self):
        return self.ds


if __name__ == "__main__":
    cid = CalDataset(is_train=False)
    dataloader = DataLoader(cid.get_ds(), batch_size=50, shuffle=True)
    for image, label in dataloader:
        print(image.shape)
        print(label)
        for i in image:
            plt.imshow(i[0].numpy())
            plt.show()
            exit(0)
