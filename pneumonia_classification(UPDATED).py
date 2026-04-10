
from __future__ import print_function #B0151878 SEGMENT 1 OF UPDATED PNEMONIOS CLASSIFICATITION

import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling, BatchNormalization
from keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
import numpy as np

# NEW IMPORTS FOR IMPROVED MODEL
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import time


batch_size = 32
num_classes = 3
epochs = 15
img_width = 224   # INCREASED IMAGE SIZE TO MATCH MOBILENETV2 INPUT REQUIREMENTS
img_height = 224
img_channels = 3
fit = False #CHNAAGED to avoid retrianing AND TIME COSUMING

# UPDATED PATHS TO YOUR LOCAL DIRECTORY
train_dir = 'C:\\Users\\jordo\\OneDrive\\Desktop\\CV-AS2\\chest_xray\\train'
test_dir  = 'C:\\Users\\jordo\\OneDrive\\Desktop\\CV-AS2\\chest_xray\\test'
val_dir   = 'C:\\Users\\jordo\\OneDrive\\Desktop\\CV-AS2\\chest_xray\\val'


