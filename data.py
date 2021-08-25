import torch
import torchvision
from torchvision import transforms,datasets


transform = transforms.Compose(
    [   
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean =(0.485,0.456,0.486),
            std=(0.229,0.224,0.225)),
    ]
)

batch_size = 4

train_set = datasets.CIFAR10(
                root="./data_dir",
                train=True,
                download=True,
                transform=transform
            )

trainloader = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
            )

test_set = datasets.CIFAR10(
                root="./data_dir",
                train=False,
                download=True,
                transform=transform
            )

testloader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
            )

