from model import classifier_net, feature_net, matchnet
from dataloader import DBcombiner, patchSet
import torch
import torchvision.transforms as transforms
import numpy as np
import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from config import DBs, DATA_DIR, TEMPS_DIR

def ParseArgs():
    """Parse input arguments.
    """
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('DATA_dir',
                        help='Feature network description.')
    
    args = parser.parse_args()
    return args


def train_one_epoch(data_loader, optimizer, model, loss_fn):
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
            last_loss = running_loss # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            # tb_x = epoch_index * len(data_loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

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

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True, num_workers=4)
        # collate_fn=torch.utils.data.collate_fn)

    dataiter = iter(data_loader)
    images, labels = dataiter.next()

    print(type(images))
    print(images.shape)
    print(labels.shape)

    # featNet = feature_net()
    # metricNet = classifier_net()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 150000

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        avg_loss = train_one_epoch(data_loader, optimizer, model, loss_fn)
        # update the learning rate
        # lr_scheduler.step()
        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)
    torch.save(model.state_dict(), '/data/p306627/models/matchnet')

    print("That's it!")

    # print(featNet)                         # what does the object tell us about itself?



if __name__ == '__main__':
    main()
