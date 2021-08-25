from pickle import decode_long
import torch
import torch.nn as nn
from torchvision.datasets import vision
from vit import *
from data import *

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using : "+ "cuda" if torch.cuda.is_available() else "cpu")



PATH = "./saved_model/vit_cifar.pth"
vision_transformer = ViT(image_size=256,patch_size=32,num_classes=10,dim=1024,depth=6,heads=16,mlp_dim=2948,dropout=0.1)
vision_transformer.load_state_dict(torch.load(PATH))
vision_transformer = vision_transformer.to(device)

correct = total = 0

def test():
    global total,correct
    with torch.no_grad():
        for data in testloader:
            img, label = data
            img = img.to(device)
            label = label.to(device)

            out = vision_transformer(img)

            _, predicted = torch.max(out.data, 1)

            total += label.size(0)
            correct += (predicted==label).sum().item()



test()
print("Finished Testing xD")
print("Accuracy : "+str(100*correct/total))







