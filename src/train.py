from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import cv2
from utils import preprocess,load_dataset,performDataAugmentation
from torch.utils.data import TensorDataset

class PhoneLocator(nn.Module):
    # define the layers
    def __init__(self):
        super(PhoneLocator, self).__init__()
        # 2 convolutional layers nn.Conv2d(in_channels,out_channels,kernel_size,stride)
        self.conv1 = nn.Conv2d(in_channels=3, 
            out_channels=6,
            kernel_size=5,
            stride=5)
        self.conv2 = nn.Conv2d(6, 64, 3, 1)
        # 2 dropout layers used for regularization
        # Randomly zero out channels with provided probability
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # 2 fully connected layers used to process features extracted by conv layers
        # nn.Linear(in_features,out_features)
        self.fc1 = nn.Linear(95232, 128)
        self.fc2 = nn.Linear(128, 2)

    # define the foward pass, including the operations between the layers
    # Operations includ ReLu activations, max pooling, flattening before the fully connected layers
    # and softmax on the output to produce a normalized (1,10) output vector
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# train the classifier for a single epoch
def train(model, device, train_loader, optimizer, epoch):
    model.train() # training  mode
    MSE = torch.nn.MSELoss()
    for batch_idx, (data, target) in enumerate(train_loader): # iterate across training dataset using batch size
        data, target = data.to(device), target.to(device) #
        optimizer.zero_grad() # set gradients to zero
        output = model(data) # get the outputs of the model
        loss = MSE(output, target)
        loss.backward() # Accumulate the gradient
        optimizer.step() # based on currently stored gradient update model params using optomizer rules
        if batch_idx % 10 == 0: # provide updates on training process
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# test the classifier
def test(args, model, device, test_loader):
    model.eval() # inference mode
    test_loss = 0
    correct = 0
    loss = torch.nn.MSELoss()
    with torch.no_grad():
        for data, target in test_loader: # load the data
            data, target = data.to(device), target.to(device)
            output = model(data) # collect the outputs
            test_loss += loss(output, target)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset) # compute the average loss

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    batch_size = 8
    learning_rate = 0.001
    gamma = 0.5
    epochs = 100
    lr_scheduler_step_size = 10
    adam_betas = (0.9,0.999)

    # attempt to use GPU if available
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123456789)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    # load dataset
    x_train,y_train,x_val,y_val,x_test = load_dataset()
    # preprocess training and validation
    x_train = preprocess(x_train)
    x_val = preprocess(x_val)
    # augment data augmentation
    x_train,y_train = performDataAugmentation(x_train,y_train)
    # load the model
    model = PhoneLocator().to(device)
    # load the optimizer and setup schedule to reduce learning rate every 10 epochs
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=adam_betas)
    scheduler = StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=gamma)

    train_dataset = (torch.FloatTensor(x_train),torch.FloatTensor(y_train))
    # create train and validatoin data loader
    train_loader = torch.utils.data.DataLoader(TensorDataset(*train_dataset),batch_size=batch_size, shuffle=True, **kwargs)

    # each iteration gather the n=test_batch_size samples and their respective labels [0,9]
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        #test(model, device, x_val,y_val)
        scheduler.step()
        torch.save(model.state_dict(), "PhoneDetector.pt")


if __name__ == '__main__':
    main()