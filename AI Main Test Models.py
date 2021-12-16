'''
This file was run once all of the best models were saved from each test.
This would do all of the calculations that we needed to determine which model was the best one.

By: Brandon Strong

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


# Setting a random seed so we all get the same 'random' numbers
np.random.seed(22)

# Names to file locations so they can be changed easily
fileName = 'model percents.txt'
fileLocation = 'D:\\model saves pool\\'

#modelLocation = 'D:\\model saves pool\\tmp\\checkpoint\\'
#modelName = 'adamaxE12085.09BEST_L1N150L2N610_12_14K18K27P110P29_2'
#extension = '.h5'
#scModelName = modelName

#datasetLocation = 'C:\\Users\\stron\\Downloads\\AI Project lung data\\chest_xray\\'
#testDataset_location = 'C:\\Users\\stron\\Downloads\\AI Project lung data\\chest_xray\\test\\'
#normalPathT = testDataset_location + 'NORMAL\\'
#pnPathT = testDataset_location + 'PNEUMONIA\\'




# This is just in case images needs to be loaded in
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
            image = cv2.resize(image, dsize=(200, 200))

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
            image = cv2.resize(image, dsize=(200, 200))

            # Make the image into gray scale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # add the image to the image array
            pneu_images.append(image)

        # Convert the image array to a numpy array
        pneu_images = np.array(pneu_images)

        # Return the labels and the array with the images in them
        return pneu_images, pneu_labels


print('Opening data...')
with open('pneumonia_data.pickle', 'rb') as f:
    (xTrain, X_test, yTrain, y_test) = pickle.load(f)

modelLocation = 'D:\\model saves pool\\tmp\\checkpoint\\'

files = np.array(os.listdir(modelLocation))

for n in tqdm(files):

    modelName = 'n'
    extension = '.h5'
    scModelName = modelName

    print('Loading the model...')
    model = load_model(modelLocation + modelName + extension)

    print('Making predictions...')
    predictions = model.predict(X_test)

    # Create array for the final predictions
    finalPredictions = []

    # Make the class names
    classnames = ['bacteria', 'normal', 'virus']

    # loop through all of the predictions for each image
    for i in predictions:

        # Reset vars
        index = 0
        iteration = 0
        biggest = np.float32(0)

        # loop through the 3 choices. Get the highest one
        for l in i:
            if l > biggest:
                biggest = l
                index = iteration
            iteration += 1

        # Add the highest predictions label to the array
        finalPredictions.append(classnames[index])


    # Make the confusion matrix with the predictions vs the actual values
    cm = confusion_matrix(y_test, finalPredictions)

    total = 0

    # Get the total for the confusion matrix
    for i in cm:
        for j in i:
            total += j

    # Do the calculations for the confusion matrix
    percent = ((cm[0][0] + cm[1][1] + cm[2][2]) / total) * 100
    sensitivity = (cm[0][0] / (cm[0][0] + cm[0][1] + cm[0][2])) * 100
    specificity = ((cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]) / (cm[1][0] + cm[1][1] + cm[1][2] + cm[2][0] + cm[2][1] +
                                                                  cm[2][2])) * 100
    ppv = (cm[0][0] / (cm[0][0] + cm[1][0] + cm[2][0])) * 100
    npv = ((cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]) / (cm[0][1] + cm[0][2] + cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2])) * 100
    accuracyBinary = ((cm[0][0] + cm[1][1] + cm[2][2] + cm[0][2] + cm[2][0]) / total) * 100
    fp = cm[0][1] + cm[2][1]

    # Open the file and write the calculated information into the file
    file = open(fileLocation + fileName, 'a')
    file.write('\nModel: ' + scModelName)
    file.write('\nAccuracy: ')
    file.write(str(percent) + '%')
    file.write('\nSensitivity: ')
    file.write(str(sensitivity) + '%')
    file.write('\nSpecificity: ')
    file.write(str(specificity) + '%')
    file.write('\nPositive Predictive Value: : ')
    file.write(str(ppv) + '%')
    file.write('\nNegative Predictive Value: ')
    file.write(str(npv) + '%')
    file.write('\nBinary Accuracy: ')
    file.write(str(accuracyBinary) + '%')
    file.write('\nNum Of False Negatives: ')
    file.write(str(fp) + '\n')
    file.close()

    # Visualize the confusion matrix
    plt.figure(figsize=(8, 8))
    plt.title('Confusion matrix')
    sns.heatmap(cm, cbar=False, xticklabels=classnames, yticklabels=classnames, fmt='d', annot=True,
                cmap=plt.cm.Blues)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    accuracyString = fileLocation + '\\Confusion Matrix\\' + scModelName + '.png'
    plt.savefig(accuracyString, bbox_inches='tight')
