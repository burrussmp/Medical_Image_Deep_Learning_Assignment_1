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
from utils import preprocess,load_dataset,performDataAugmentation,reshapeInput
from torch.utils.data import TensorDataset
import math
class PhoneLocator(nn.Module):
    # define the layers
    def __init__(self):
        super(PhoneLocator, self).__init__()
        # 2 convolutional layers nn.Conv2d(in_channels,out_channels,kernel_size,stride)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128,kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256,kernel_size=3)
        self.dropout1 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(32256, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)
    # define the foward pass, including the operations between the layers
    # Operations includ ReLu activations, max pooling, flattening before the fully connected layers
    # and softmax on the output to produce a normalized (1,10) output vector
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 4)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 8)  

        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.dropout1(x)
        output = self.fc3(x)
        return output
# train the classifier for a single epoch
def train(model, device, train_loader, optimizer, epoch):
    model.train() # training  mode
    criteria = torch.nn.MSELoss()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader): # iterate across training dataset using batch size
        data, target = data.to(device), target.to(device) #
        optimizer.zero_grad() # set gradients to zero
        output = model(data) # get the outputs of the model
        loss = criteria(output, target)
        total_loss += loss
        loss.backward() # Accumulate the gradient
        optimizer.step() # based on currently stored gradient update model params using optomizer rules
        if batch_idx % 10 == 0: # provide updates on training process
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss

# test the classifier
def validate(model, device, validation_loader):
    model.eval() # inference mode
    test_loss = 0
    correct = 0
    criteria = torch.nn.MSELoss()
    #absolute_error = torch.nn.L1Loss()
    with torch.no_grad():
        for data, target in validation_loader: # load the data
            data, target = data.to(device), target.to(device)
            output = model(data) # collect the outputs
            test_loss += criteria(output, target)  # sum up batch loss
            distance = torch.dist(target,output)
            print(output)
            print(target)
            correct += distance.lt(torch.tensor(0.05)).sum().item()
    test_loss /= len(validation_loader.dataset) # compute the average loss
    print('Test set: Average loss: {:.4f}, Number within range: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))
    return test_loss


def main():
    # Training settings
    batch_size = 16
    learning_rate = 0.001
    gamma = 0.5
    epochs = 100
    lr_scheduler_step_size = 10
    adam_betas = (0.9,0.999)
    pathToModel = './PhoneDetector.pt'
    restart = True

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
    #x_train,y_train = performDataAugmentation(x_train,y_train)
    #x_val,y_val = performDataAugmentation(x_val,y_val)
    x_train = reshapeInput(x_train)
    x_val = reshapeInput(x_val)
    # load the model
    model = PhoneLocator().to(device)
    # load the optimizer and setup schedule to reduce learning rate every 10 epochs
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=adam_betas)
    scheduler = StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=gamma)

    train_dataset = (torch.FloatTensor(x_train),torch.FloatTensor(y_train))
    validation_dataset = (torch.FloatTensor(x_val),torch.FloatTensor(y_val))
    # create train and validatoin data loader
    train_loader = torch.utils.data.DataLoader(TensorDataset(*train_dataset),batch_size=batch_size, shuffle=True, **kwargs)
    validation_loader = torch.utils.data.DataLoader(TensorDataset(*validation_dataset), shuffle=True, **kwargs)
    # load model if path exists
    if os.path.isfile(pathToModel) and not restart:
        model.load_state_dict(torch.load(pathToModel))
    # each iteration gather the n=test_batch_size samples and their respective labels [0,9]
    best_loss = math.inf
    train_loss_save = np.zeros((epochs))
    val_loss_save = np.zeros((epochs))
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        val_loss = validate(model, device, validation_loader)
        if (use_cuda):
            train_loss_save[epoch-1] = train_loss.cpu().data.numpy()
            val_loss_save[epoch-1] = val_loss.cpu().data.numpy()
        else:
            train_loss_save[epoch-1] = train_loss.data.numpy()
            val_loss[epoch-1] = val_loss.data.numpy()
        if (val_loss < best_loss):
            print('Loss improved from ', best_loss, 'to',val_loss,': Saving new model to',pathToModel)
            best_loss = val_loss
            torch.save(model.state_dict(), pathToModel)
        scheduler.step()
        np.save('/content/drive/My Drive/MedicalImageAssignments/val_loss.npy',val_loss_save)
        np.save('/content/drive/My Drive/MedicalImageAssignments/train_loss.npy',train_loss_save)

if __name__ == '__main__':
    main()