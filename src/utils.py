import numpy as np
import cv2
import os
from config import *

def load_dataset(img_size=(326,490,3)):
    assert os.path.isfile(PATH_TO_LABELS), \
        print('path to labels.txt is invalid.')
    data = np.loadtxt(PATH_TO_LABELS,dtype=str,delimiter=' ')
    paths = data[:,0]
    output = data[:,1::].astype(np.float32)
    training_size = len(os.listdir(PATH_TO_TRAIN_FOLDER))
    validation_size = len(os.listdir(PATH_TO_VALIDATION_FOLDER))
    test_size = len(os.listdir(PATH_TO_TEST_FOLDER))
    x_train = np.zeros((training_size,img_size[0],img_size[1],img_size[2]))
    y_train = np.zeros((training_size,2))
    x_val = np.zeros((validation_size,img_size[0],img_size[1],img_size[2]))
    y_val = np.zeros((validation_size,2))
    x_test = np.zeros((test_size,img_size[0],img_size[1],img_size[2]))
    valIdx = 111
    k = 0
    j = 0
    for i in range(paths.shape[0]):
        curPath = paths[i]
        if (int(curPath.replace('.jpg','')) >= valIdx):
            path = os.path.join(PATH_TO_VALIDATION_FOLDER,curPath)
            typeData = 'validation'
        else:
            path = os.path.join(PATH_TO_TRAIN_FOLDER,curPath)
            typeData = 'train'
        if not os.path.isfile(path):
            print('not existing')
            continue
        img = cv2.imread(path)
        if (typeData == 'validation'):
            x_val[j] = img
            y_val[j] = output[i]
            j += 1
        else:
            x_train[k] = img
            y_train[k] = output[i]
            k += 1
    i = 0
    for test_file in os.listdir(PATH_TO_TEST_FOLDER):
        path = os.path.join(PATH_TO_TEST_FOLDER,test_file)
        img = cv2.imread(path)
        x_test[i] = img
        i += 1
    return x_train,y_train,x_val,y_val,x_test

def preprocess(X):
    X /= 255
    X *= 2
    X -= 1
    return X

def AugmentFlipImage(X,Y):
    x_flip_horizontal = np.flip(X,axis=2)
    y_flip_horizontal = np.copy(Y)
    y_flip_horizontal[:,0] = 1 - y_flip_horizontal[:,0]
    x_flip_vertical = np.flip(X,axis=1)
    y_flip_vertical = np.copy(Y)
    y_flip_vertical[:,1] = 1 - y_flip_vertical[:,1]
    x_flip_vertical_horizontal = np.flip(x_flip_vertical,axis=2)
    y_flip_vertical_horizontal = np.copy(y_flip_vertical)
    y_flip_vertical_horizontal[:,0] = 1 - y_flip_vertical_horizontal[:,0]
    X = np.concatenate((X,x_flip_horizontal,x_flip_vertical,x_flip_vertical_horizontal),axis=0)
    Y = np.concatenate((Y,y_flip_horizontal,y_flip_vertical,y_flip_vertical_horizontal),axis=0)
    return X,Y

def AugmentBrightnessRandomly(X,Y,br=0.25):
    x_bright = np.zeros_like(X)
    for i in range(X.shape[0]):
        image = X[i]
        image = image.astype(np.uint8)
        rand_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        rand_bright = br + np.random.uniform()
        rand_image[:,:,2] = rand_image[:,:,2]*rand_bright
        rand_image = cv2.cvtColor(rand_image, cv2.COLOR_HSV2RGB)
        x_bright[i] = rand_image.astype(np.float32)
    X = np.concatenate((X,x_bright),axis=0)
    Y = np.concatenate((Y,Y),axis=0)
    return X,Y


def performDataAugmentation(X,Y):
    X,Y = AugmentFlipImage(X,Y)
    X,Y = AugmentBrightnessRandomly(X,Y,br=0.25)
    X = np.swapaxes(X, 1, 3)
    return X,Y