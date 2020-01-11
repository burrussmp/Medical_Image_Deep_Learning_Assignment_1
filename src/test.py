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
from train import PhoneLocator


def main():
    parser = argparse.ArgumentParser(description='Assignment 1')
    parser.add_argument("files",nargs="*")
    args = parser.parse_args()
    if (len(args.files)==0):

    else:
        for file in files:
            assert os.path.isfile(file),\
                print('SKIPPING: File does not exist: ', file)
            
    # attempt to use GPU if available
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123456789)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    x_train,y_train,x_val,y_val,x_test = load_dataset()


if __name__ == '__main__':
    main()