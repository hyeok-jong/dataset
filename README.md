# dataset


```
import torchvision
import torch
data_set = torchvision.datasets.CIFAR10(root = './data', train=True,
                                        download=True, transform=None)

# Split the dataset into training and validation sets
train_len = int(len(data_set) * 0.5)
valid_len = len(data_set) - train_len
train_set, valid_set = torch.utils.data.random_split(data_set, [train_len, valid_len])


train_set.dataset.transforms = 1
valid_set.dataset.transforms

```
