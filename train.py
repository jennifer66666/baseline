from dataset import LandmarksDataset
from model import *
import torch
import torch.optim as optim
import random
from torch.utils.tensorboard import SummaryWriter

def main(split,batch_size,epochs,log_every_batches,name):
    #--------------------------------train config----------------------------
    # for regression
    device = torch.device("cuda:0")
    print(device)
    criterion = nn.L1Loss()
    model = Model()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # For tensorboard visualization
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/'+name)
    #---------------------------------prepare ds-----------------------------
    fullset = LandmarksDataset(root_dir="dataset/train_FAN")
    train_size = int(split * len(fullset))
    test_size = len(fullset) - train_size
    trainset, testset = torch.utils.data.random_split(fullset, [train_size, test_size])
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=4)
    testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=4)
    #---------------------------------start train-----------------------------
    for epoch in range(epochs):
        # running loss for checking only, loss for forward and backward
        running_loss = 0.0
        for i,data in enumerate(trainloader,0):
            inputs,labels = data[0].to(device),data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # log every designated mini-batches
            if i % log_every_batches == log_every_batches-1:    
                # log the running loss for tensorboard
                writer.add_scalar('training loss',
                                running_loss / (log_every_batches*batch_size),
                                epoch * len(trainloader) + i)
                running_loss = 0.0
        # do test after every epoch finish
        loss_test = 0.0
        for i,data in enumerate(testloader,0):
            inputs,labels = data[0].to(device),data[1].to(device)
            outputs = model(inputs.float())
            loss=criterion(outputs,labels)
            loss_test += loss.item()
        writer.add_scalar('test loss',
                                loss_test / (test_size),
                                epoch * len(trainloader))
    print('Finished Training')
    PATH = './regression_net.pth'
    torch.save(model.state_dict(),PATH)

if __name__ == "__main__":
    split = 0.8
    batch_size = 10
    epochs = 30
    log_every_batches = 100
    name = "experiment_4"
    main(split,batch_size,epochs,log_every_batches,name)