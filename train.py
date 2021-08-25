import torch
import torch.nn as nn
from vit import *
from data import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using : "+ "cuda" if torch.cuda.is_available() else "cpu")

vision_transformer = ViT(image_size=256,patch_size=32,num_classes=10,dim=1024,depth=6,heads=16,mlp_dim=2948,dropout=0.1)
vision_transformer = vision_transformer.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(vision_transformer.parameters(),lr=0.001,momentum=0.9)

def train():
    for epoch in range(5):
        running_loss=0.0

        for i,data in enumerate(trainloader,0):
            img,label=data
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            #print(img.shape)
            #print(labels.shape)
            out = vision_transformer(img)
            
            loss = loss_fn(out,label)
            optimizer.step()

            running_loss += loss.item()

            if i%100==99:
                print(f"Loss = {running_loss/2000} @Epoch = {epoch+1}")


train()
PATH = "./saved_model/vit_cifar.pth"
torch.save(vision_transformer.state_dict(),PATH)
print("FINISHED TRAINING and SAVED")