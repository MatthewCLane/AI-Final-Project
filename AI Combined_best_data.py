'''

This is to find the best model with all of the best settings that we have aquired.
The program runs 100 times and saves the best model for each iteration. It also saves
the confusion matrix, loss graph, and accuracy graph for each iteration.

Edited By: Brandon Strong

'''

import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from keras.models import Model, load_model
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf

# Path to the train data
normalPath = 'C:\\Users\\stron\\Downloads\\AI Project lung data - Copy\\chest_xray\\train\\NORMAL\\'
pnPath = 'C:\\Users\\stron\\Downloads\\AI Project lung data - Copy\\chest_xray\\train\\PNEUMONIA\\'

# Path to the test data
normalPathT = 'C:\\Users\\stron\\Downloads\\AI Project lung data - Copy\\chest_xray\\test\\NORMAL\\'
pnPathT = 'C:\\Users\\stron\\Downloads\\AI Project lung data - Copy\\chest_xray\\test\\PNEUMONIA\\'

# Setting a random seed so we all get the same 'random' numbers
np.random.seed(22)

fileOutput = 'C:\\Users\\stron\\OneDrive\\Desktop\\Project\combined models\\info.txt'


# load the images into memory
def load_images(path):
    # Checking to see what kind of images we have. Normal xrays or phenomena

    # We are dealing with normal xrays
    if 'NORMAL' in path:
        # Get an array with a list of all the files
        norm_files = np.array(os.listdir(path))

        # Make an array for the labels
        norm_labels = np.array(['normal'] * len(norm_files))

        # Declare array for the images
        norm_images = []
        for image in tqdm(norm_files):
            image = cv2.imread(path + image)
            image = cv2.resize(image, dsize=(100, 100))

            # Convert the image to gray scale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Make a list of all the modified images
            norm_images.append(image)

        # Make the list a numpy array so it's more efficient (and so we can work with it easier)
        norm_images = np.array(norm_images)

        # Return all of the modified images, as well as the classification of the image
        return norm_images, norm_labels

    # We are dealing with pneumonia images
    else:
        # Get an array with a list of all the files
        pneu_files = np.array(os.listdir(path))

        # Make an array for the labels
        pneu_labels = np.array([pneu_file.split('_')[1] for pneu_file in pneu_files])

        # Declare array for the images
        pneu_images = []

        # Loop through all of the images (display a progress bar)
        for image in tqdm(pneu_files):
            # Read the image from the path
            image = cv2.imread(path + image)

            # Resize the image
            image = cv2.resize(image, dsize=(100, 100))

            # Make the image into gray scale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # add the image to the image array
            pneu_images.append(image)

        # Convert the image array to a numpy array
        pneu_images = np.array(pneu_images)

        # Return the labels and the array with the images in them
        return pneu_images, pneu_labels


#
# IF THERE ISN'T A '.PICKLE' FILE THEN UNCOMMENT THESE LINES!!!!!
#

# load the images into memory start with NORMAL
normalImages, normalLabels = load_images(normalPath)

#load the images into memory now do PNEUMONIA
pnImages, pnLabels = load_images(pnPath)

xTrain = np.append(normalImages, pnImages, axis=0)
yTrain = np.append(normalLabels, pnLabels)

#
# IF THERE ISN'T A '.PICKLE' FILE THEN UNCOMMENT THESE LINES!!!!!
#

# Read in the test images now
norm_images_test, norm_labels_test = load_images(normalPathT)
pneu_images_test, pneu_labels_test = load_images(pnPathT)

# Save the test data into np arrays
X_test = np.append(norm_images_test, pneu_images_test, axis=0)
y_test = np.append(norm_labels_test, pneu_labels_test)

# Use this to save variables (Converting object data into byte data)
# (serializing and deserializing a Python object structure)
with open('pneumonia_data.pickle_100x100', 'wb') as f:
    pickle.dump((xTrain, X_test, yTrain, y_test), f)# Use this to load variables


#
# IF THERE ISN'T A '.PICKLE' FILE THEN COMMENT THESE LINES!!!!!
#

#with open('pneumonia_data.pickle_50x50', 'rb') as f:
#    (xTrain, X_test, yTrain, y_test) = pickle.load(f)

# Give the y Train and test set a new axis. The OneHotEncoder will be expecting another axis
yTrain = yTrain[:, np.newaxis]
y_test = y_test[:, np.newaxis]

# Initialize the encoder
one_hot_encoder = OneHotEncoder(sparse=False)

# Change the data to a different format
y_train_one_hot = one_hot_encoder.fit_transform(yTrain)
y_test_one_hot = one_hot_encoder.transform(y_test)

# Add a color variable to the x train and test sets
xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1], xTrain.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Create more training samples by changing the photos. (zoom in, rotate image, etc)
datagen = ImageDataGenerator(
    rotation_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1)

# Calculate any statistics required to actually perform the transforms to your image data
datagen.fit(xTrain)

# Configure the batch size and prepare the data generator and get batches of images
train_gen = datagen.flow(xTrain, y_train_one_hot, batch_size=32)

# The numbers that we will use for layer 1's input
input1 = Input(shape=(xTrain.shape[1], xTrain.shape[2], 1))

#
# Neural network, too complicated to explain look at documentation for 'Conv2D', 'MaxPool2D'
# https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
# https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
#
counter = 12
counter2 = 14
iteration = 0
iteration2 = 0
reset = 0


while iteration < 100:

    iteration += 1

    file = open(fileOutput, 'a')
    cnn = Conv2D(counter, (8, 8), activation='relu', strides=(1, 1),
                 padding='same')(input1)
    cnn = Conv2D(counter2, (8, 8), activation='relu', strides=(1, 1),
                 padding='same')(cnn)
    cnn = MaxPool2D((8, 8))(cnn)

    cnn = Conv2D(counter, (7, 7), activation='relu', strides=(1, 1),
                 padding='same')(cnn)
    cnn = Conv2D(counter2, (7, 7), activation='relu', strides=(1, 1),
                 padding='same')(cnn)
    cnn = MaxPool2D((9, 9))(cnn)

    #
    #  Neural network look at documentation for 'Flatten', 'Dense'
    # https://stackoverflow.com/questions/43237124/what-is-the-role-of-flatten-in-keras
    # https://keras.io/api/layers/core_layers/dense/
    #
    cnn = Flatten()(cnn)
    cnn = Dense(150, activation='relu')(cnn)
    cnn = Dense(610, activation='relu')(cnn)
    output1 = Dense(3, activation='softmax')(cnn)

    #
    # Model groups layers into an object with training and inference features.
    # https://keras.io/api/models/model/#model-class
    #
    model = Model(inputs=input1, outputs=output1)

    # https://keras.io/api/models/model_training_apis/#compile-method
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['acc'])
    checkpoint_filepath = 'C:\\Users\\stron\\OneDrive\\Desktop\\Project\\combined models\\tmp\\checkpoint\\adamaxE12085.09_50x50_L1N150L2N610_' + \
                          str(counter) + '_' + str(counter2) + 'K18' + 'K27' + 'P1' + str(8) + 'P2' + str(9) +\
                     '_' + str(iteration) + '.h5'

    earlyStopping = EarlyStopping(monitor='val_acc', patience=130, verbose=0, mode='max')
    mcp_save = ModelCheckpoint(checkpoint_filepath, save_best_only=True, monitor='val_acc', mode='max')
    #reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4,
    #                                   mode='min')

    # https://www.geeksforgeeks.org/keras-fit-and-keras-fit_generator/
    history = model.fit_generator(train_gen, epochs=130,
                                  validation_data=(X_test, y_test_one_hot), callbacks=[earlyStopping, mcp_save])
    model.save('C:\\Users\\stron\\OneDrive\\Desktop\\Project\\combined models\\adamE12085.09_50x50_L1N150L2N610_' + str(counter) + '_' + str(counter2) +
               'K17' + 'K28' + 'P1' + str(8) + 'P2' + str(9) + '_' + \
                     str(iteration) + '.h5')

    # Get the predictions from the neural network
    predictions = model.predict(X_test)
    predictions = one_hot_encoder.inverse_transform(predictions)

    # Make the confusion matrix with the predictions vs the actual values
    cm = confusion_matrix(y_test, predictions)

    percent = ((cm[0][0] + cm[1][1] + cm[2][2]) / 624) * 100

    file.write('\nAccuracy: ')
    file.write(str(percent))
    file.write('\nP1: ')
    file.write(str(8))
    file.write('\nP2: ')
    file.write(str(9))
    file.write('\nIteration: ')
    file.write(str(iteration))
    file.write('\n')

    # Visualize the confusion matrix
    classnames = ['bacteria', 'normal', 'virus']
    plt.figure(figsize=(8, 8))
    plt.title('Confusion matrix')
    sns.heatmap(cm, cbar=False, xticklabels=classnames, yticklabels=classnames, fmt='d', annot=True,
                cmap=plt.cm.Blues)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    accuracyString = 'C:\\Users\\stron\\OneDrive\\Desktop\\Project\\combined models\\Confusion Matrix\\adamE12085.09_50x50confusionMatrix_L1N150L2N610_' + \
                     str(counter) \
                     + '_' + str(counter2) + 'K17' + 'K28' + 'P1' + str(8) + 'P2' + str(9) + '_' + \
                     str(iteration) + '.png'
    plt.savefig(accuracyString, bbox_inches='tight')

    # Plot the first values
    plt.figure(figsize=(8, 6))
    plt.title('Accuracy scores')
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['acc', 'val_acc'])
    accuracyString = 'C:\\Users\\stron\\OneDrive\\Desktop\\Project\\combined models\\Accuracy\\adamE12085.09_50x50confusionMatrix_L1N150L2N610_' + \
                     str(counter) \
                     + '_' + str(counter2) + 'K17' + 'K28' + 'P1' + str(8) + 'P2' + str(9) + '_' + \
                     str(iteration) + '.png'
    plt.savefig(accuracyString, bbox_inches='tight')
    # plt.show()

    # Plot the second values
    plt.figure(figsize=(8, 6))
    plt.title('Loss value')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    accuracyString = 'D:\\best model\\loss\\adamE12085.09_50x50confusionMatrix_L1N150L2N610_' + \
                     str(counter) \
                     + '_' + str(counter2) + 'K17' + 'K28' + 'P1' + str(8) + 'P2' + str(9) + '_' + \
                     str(iteration) + '.png'
    plt.savefig(accuracyString, bbox_inches='tight')

    reset += 1

    # If reset = 2 then delete the variables
    if reset % 2 == 0:
        del cnn
        del model
        del output1

    file.close()
