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



Thus   



```

import torch
import torchvision
import torchvision.transforms as transforms

# Define the transformations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load the training data
data_root_dir = './data'
data_set = torchvision.datasets.CIFAR10(root=data_root_dir, train=True, download=True, transform=train_transform)

# Split the training data into training and validation sets
train_ratio = 0.8
train_size = int(len(data_set) * train_ratio)
valid_size = len(data_set) - train_size

train_set, valid_set = torch.utils.data.random_split(data_set, [train_size, valid_size])

# Apply the test transformation to the validation set
valid_set = torch.utils.data.Subset(valid_set.dataset, valid_set.indices)
valid_set.dataset.transform = test_transform

# Load the test data
test_set = torchvision.datasets.CIFAR10(root=data_root_dir, train=False, download=True, transform=test_transform)

# Define the dataloaders
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)


```
