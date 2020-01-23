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
from train import transferResNet18
import matplotlib.pyplot as plt
from config import *

def createImages(model,use_cuda):
    pathToTest = PATH_TO_TEST_FOLDER
    for file in os.listdir(pathToTest):
        orig_img = cv2.imread(os.path.join(pathToTest,file)).astype(np.float32)
        img = np.expand_dims(preprocess(np.copy(orig_img)),axis=0)
        img = reshapeInput(img)
        prediction = model(torch.tensor(img))
        if (use_cuda):
            result = prediction.cpu().data.numpy()[0]
        else:
            result = prediction.data.numpy()[0]
        print('File:',file,'\nResult:',np.round(result,4),'\n')
        y = int(result[1]*orig_img.shape[0])
        x = int(result[0]*orig_img.shape[1])
        drawn_circle = cv2.circle(orig_img, (x,y), 4, (0,0,255), -1)
        #cv2.imwrite('./resources/'+file,drawn_circle.astype(np.uint8))
        # cv2.imshow('Located phone',drawn_circle.astype(np.uint8))
        # cv2.waitKey(0)
    # plot the training and validation loss
    val_loss = np.load(os.path.join(PATH_TO_VALIDATION_LOSS,'val_loss.npy'))
    train_loss = np.load(os.path.join(PATH_TO_TRAIN_LOSS,'train_loss.npy'))
    epochs = np.arange(len(val_loss))
    plt.figure()
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(epochs,val_loss,label='Validation Loss')
    plt.plot(epochs,train_loss,label='Training Loss')
    plt.legend()
    plt.show()

def main():
    # parameters
    pathToModel = './src/PhoneDetector.pt'
    parser = argparse.ArgumentParser(description='Assignment 1')
    parser.add_argument("files",nargs="*")
    args = parser.parse_args()

    # attempt to use GPU if available
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123456789)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    model = transferResNet18()
    model.eval() 
    #model = PhoneLocator().to(device)
    if (use_cuda):
        model.cuda()       
    if os.path.isfile(pathToModel):
        if not use_cuda:
            model.load_state_dict(torch.load(pathToModel,map_location='cpu'))
        else:
            model.load_state_dict(torch.load(pathToModel))
    if (len(args.files)==0):
        createImages(model,use_cuda)
    else:
        for file in args.files:
            assert os.path.isfile(file),\
                print('SKIPPING: File does not exist: ', file)
            img = cv2.imread(file).astype(np.float32)
            img = np.expand_dims(preprocess(img),axis=0)
            img = reshapeInput(img)
            prediction = model(torch.tensor(img))
            if (use_cuda):
                result = prediction.cpu().data.numpy()
            else:
                result = prediction.data.numpy()
            rounded = np.round(result[0],4)
            print(rounded[0],rounded[1])

if __name__ == '__main__':
    main()