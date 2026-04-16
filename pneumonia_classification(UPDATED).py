
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



#SEGMENT 2

    # RESCALING LAYER NORMALISES PIXEL VALUES FROM 0-255 TO 0-1
    # THIS IS REQUIRED BEFORE PASSING IMAGES INTO MOBILENETV2
    rescale = tf.keras.layers.Rescaling(1.0 / 255)

    # DATA AUGMENTATION LAYER RANDOMLY TRANSFORMS IMAGES DURING TRAINING
    # THIS HELPS THE MODEL GENERALISE BETTER AND REDUCES OVERFITTING
    # ONLY APPLIED DURING TRAINING NOT DURING VALIDATION OR TESTING
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ])

    # PREPROCESS DATASETS WITH RESCALING
    # AUGMENTATION ONLY GOES ON TRAINING DATA
    train_ds_processed = train_ds.map(lambda x, y: (rescale(data_augmentation(x, training=True)), y))
    val_ds_processed   = val_ds.map(lambda x, y: (rescale(x), y))
    test_ds_processed  = test_ds.map(lambda x, y: (rescale(x), y))

    # CACHE AND PREFETCH FOR FASTER TRAINING BY LOADING DATA IN BACKGROUND
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds_processed = train_ds_processed.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds_processed   = val_ds_processed.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds_processed  = test_ds_processed.cache().prefetch(buffer_size=AUTOTUNE)

    # LOAD MOBILENETV2 PRETRAINED ON IMAGENET AS THE BASE MODEL
    # THIS IS CALLED TRANSFER LEARNING - WE REUSE FEATURES LEARNED FROM MILLIONS OF IMAGES
    # INCLUDE TOP IS FALSE SO WE REMOVE THE ORIGINAL CLASSIFICATION HEAD AND ADD OUR OWN
    base_model = MobileNetV2(
        input_shape=(img_height, img_width, img_channels),
        include_top=False,
        weights='imagenet'
    )

    # FREEZE ALL LAYERS IN THE BASE MODEL SO THEY DO NOT CHANGE DURING INITIAL TRAINING
    # WE ONLY WANT TO TRAIN OUR NEW CLASSIFICATION HEAD FIRST
    base_model.trainable = False

    # BUILD THE FULL MODEL BY ADDING OUR CUSTOM HEAD ON TOP OF MOBILENETV2
    inputs  = tf.keras.Input(shape=(img_height, img_width, img_channels))
    x       = base_model(inputs, training=False)

    # GLOBALAVERAGEPOOLING REDUCES SPATIAL DIMENSIONS TO A SINGLE VECTOR PER IMAGE
    x       = layers.GlobalAveragePooling2D()(x)

    # DENSE LAYER LEARNS TO COMBINE THE EXTRACTED FEATURES
    x       = layers.Dense(256, activation='relu')(x)

    # BATCHNORM NORMALISES ACTIVATIONS WHICH SPEEDS UP TRAINING AND IMPROVES STABILITY
    x       = layers.BatchNormalization()(x)

    # DROPOUT RANDOMLY TURNS OFF NEURONS DURING TRAINING TO PREVENT OVERFITTING
    x       = layers.Dropout(0.4)(x)

    x       = layers.Dense(128, activation='relu')(x)
    x       = layers.Dropout(0.3)(x)

    # FINAL OUTPUT LAYER HAS ONE NODE PER CLASS WITH SOFTMAX FOR MULTI CLASS CLASSIFICATION
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.summary()

    # COMPILE THE MODEL WITH ADAM OPTIMISER AND SPARSE CATEGORICAL CROSSENTROPY
    # SINCE WE HAVE 3 CLASSES AND INTEGER LABELS
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )

    # EARLY STOPPING STOPS TRAINING WHEN VALIDATION LOSS STOPS IMPROVING
    # RESTORE BEST WEIGHTS RELOADS THE WEIGHTS FROM THE BEST EPOCH
    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True
    )

    # MODEL CHECKPOINT SAVES THE BEST MODEL TO DISK DURING TRAINING
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        "pneumonia.keras",
        save_freq='epoch',
        save_best_only=True
    )

    # REDUCE LEARNING RATE WHEN VALIDATION LOSS PLATEAUS TO FINE TUNE WEIGHTS
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1
    )


#SEGMENT 3 STARTING FROM IF STATMENT REPORTINFG TRIANING TIME

    if fit:
        # RECORD START TIME SO WE CAN REPORT HOW LONG TRAINING TOOK
        start_time = time.time()

        print("\nPHASE 1 TRAINING THE CLASSIFICATION HEAD WITH FROZEN BASE MODEL")
        history = model.fit(
            train_ds_processed,
            validation_data=val_ds_processed,
            epochs=epochs,
            class_weight=class_weight_dict,
            callbacks=[earlystop_callback, save_callback, lr_callback]
        )

        # PHASE 2 FINE TUNING - UNFREEZE TOP LAYERS OF MOBILENETV2 AND TRAIN AT LOW LEARNING RATE
        # THIS ALLOWS THE PRETRAINED FEATURES TO ADAPT SLIGHTLY TO OUR SPECIFIC DATASET
        print("\nPHASE 2 FINE TUNING TOP LAYERS OF MOBILENETV2")
        base_model.trainable = True

        # ONLY UNFREEZE THE LAST 30 LAYERS TO AVOID DESTROYING EARLY LEARNED FEATURES
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        # RECOMPILE WITH A MUCH LOWER LEARNING RATE FOR FINE TUNING
        # HIGH LEARNING RATE HERE WOULD DESTROY THE PRETRAINED WEIGHTS
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=1e-5),
            metrics=['accuracy']
        )

        history_fine = model.fit(
            train_ds_processed,
            validation_data=val_ds_processed,
            epochs=10,
            class_weight=class_weight_dict,
            callbacks=[earlystop_callback, save_callback, lr_callback]
        )

        # CALCULATE AND PRINT TOTAL TRAINING TIME FOR THE REPORT
        end_time = time.time()
        training_minutes = (end_time - start_time) / 60
        print(f"\nTOTAL TRAINING TIME: {training_minutes:.1f} MINUTES")

    else:
        # LOAD PREVIOUSLY SAVED MODEL IF FIT IS SET TO FALSE
        model = tf.keras.models.load_model("pneumonia.keras")

    # EVALUATE THE MODEL ON THE TEST SET
    print("\nEVALUATING ON TEST SET")
    score = model.evaluate(test_ds_processed, batch_size=batch_size)
    print('Test loss:    ', score[0])
    print('Test accuracy:', score[1])

    # GET ALL PREDICTIONS AND TRUE LABELS FROM THE TEST SET FOR DETAILED METRICS
    all_preds  = []
    all_labels = []

