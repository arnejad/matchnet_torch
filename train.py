from model import classifier_net, feature_net, matchnet
from dataloader import DBcombiner, patchSet
import torch
import torchvision.transforms as transforms
from torch.utils.data import random_split

import numpy as np
import os
import time

from config import DBs, DATA_DIR, TEMPS_DIR, OUT_DIR


def train_one_epoch(epoch, data_loader, optimizer, model, loss_fn):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(data_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss/100 # loss per batch
            print(f'[e:{epoch + 1}, b:{i + 1:5d}] loss: {running_loss / 100:.8f}')

            # tb_x = epoch_index * len(data_loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.0

    return last_loss


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    if not os.path.exists(TEMPS_DIR+"All_matches.csv"):  #read from saved files if already available
        DBcombiner(DBs)
    # else:
    #     matchSet = np.genfromtxt(DATA_DIR+"All_matches.csv", delimiter=',')
    #     nonmatchSet = np.genfromtxt(DATA_DIR+"All_nonmatches.csv", delimiter=',')

    #check the model
    model = matchnet().to(device)
    print(model)
        
    transform = transforms.Compose([    
        transforms.ToTensor(),
        transforms.Normalize(mean=128,std=160)
    ])


    dataset = patchSet(TEMPS_DIR, transforms=transform)

    
    train_set_size = int(len(dataset) * 0.7)
    valid_set_size = len(dataset) - train_set_size
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_set_size, valid_set_size])

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, num_workers=4)    

    vaild_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=128, num_workers=4) 

    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    print(type(images))
    print(images.shape)
    print(labels.shape)


    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 150000
    train_losses = np.array([])
    vldn_losses = np.array([])
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        last_train_loss = train_one_epoch(epoch, train_loader, optimizer, model, loss_fn)
        train_losses = np.append(train_losses, last_train_loss)

        valid_loss = 0.0


        for data, labels in vaild_loader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
        
            
            target = model(data)# Forward Pass
            loss = loss_fn(target,labels) # Find the Loss
            valid_loss += loss.item() # Calculate Loss
        
        vloss = valid_loss / len(vaild_loader)
        vldn_losses = np.append(vldn_losses, vloss)

        print(f'Epoch {epoch+1} \t\t Training Loss: {last_train_loss} \t\t Validation Loss: {vloss}')

        np.savetxt(OUT_DIR+"trainLosses.csv", train_losses, delimiter=",")
        np.savetxt(OUT_DIR+"validationLosses.csv", vldn_losses, delimiter=",")

        if epoch % 5 == 0:
            print("saving model")
            torch.save(model.state_dict(), OUT_DIR+"matchnet-last")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(), OUT_DIR+"matchnet-"+timestr)

    print("That's it!")

    # print(featNet)                         # what does the object tell us about itself?



if __name__ == '__main__':
    main()
