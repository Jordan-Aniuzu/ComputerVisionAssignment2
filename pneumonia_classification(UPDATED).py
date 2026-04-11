
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

#FINISHING OF SEGMENT 1 
with tf.device('/cpu:0'):

    # CREATE TRAINING AND VALIDATION DATASETS FROM THE TRAINING DIRECTORY
    train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=123,
        validation_split=0.2,
        subset='both',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)

    # CREATE A SEPARATE TEST DATASET FROM THE TEST DIRECTORY
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=None,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=False)  # SHUFFLE OFF FOR TEST SET SO PREDICTIONS STAY IN ORDER

    class_names = train_ds.class_names
    print('Class Names: ', class_names)
    num_classes = len(class_names)

    # SHOW SAMPLE IMAGES FROM THE TRAINING SET TO CHECK DATA IS LOADING CORRECTLY
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(2):
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i].numpy()])
            plt.axis("off")
    plt.show()

    # CHECK HOW MANY IMAGES ARE IN EACH CLASS TO SEE IF DATASET IS BALANCED
    print("\nCHECKING CLASS DISTRIBUTION IN TRAINING SET")
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            print(f"  {class_name}: {count} images")

    # CALCULATE CLASS WEIGHTS TO HANDLE IMBALANCED DATASET
    # THE PNEUMONIA CLASS HAS FAR MORE SAMPLES THAN NORMAL
    # CLASS WEIGHTS PENALISE THE MODEL MORE FOR GETTING MINORITY CLASSES WRONG
    class_counts = {}
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            class_counts[class_name] = len(os.listdir(class_path))

    total_samples = sum(class_counts.values())
    class_weight_dict = {}
    for i, class_name in enumerate(class_names):
        class_weight_dict[i] = total_samples / (num_classes * class_counts.get(class_name, 1))

    print("\nCLASS WEIGHTS APPLIED TO HANDLE IMBALANCE")
    print(class_weight_dict)

    # RESCALING LAYER NORMALISES PIXEL VALUES FROM 0-255 TO 0-1
    # THIS IS REQUIRED BEFORE PASSING IMAGES INTO MOBILENETV2
    rescale = tf.keras.layers.Rescaling(1.0 / 255)
