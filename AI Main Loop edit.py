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

# Path to the train data
normalPath = 'C:\\Users\\stron\\Downloads\\AI Project lung data\\chest_xray\\train\\NORMAL\\'
pnPath = 'C:\\Users\\stron\\Downloads\\AI Project lung data\\chest_xray\\train\\PNEUMONIA\\'

# Path to the test data
normalPathT = 'C:\\Users\\stron\\Downloads\\AI Project lung data\\chest_xray\\test\\NORMAL\\'
pnPathT = 'C:\\Users\\stron\\Downloads\\AI Project lung data\\chest_xray\\test\\PNEUMONIA\\'

fileOutput = 'C:\\Users\\asus\\Desktop\\samples.txt'

# Setting a random seed so we all get the same 'random' numbers
np.random.seed(22)

epochs = 20


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

#
# IF THERE ISN'T A '.PICKLE' FILE THEN UNCOMMENT THESE LINES!!!!!
#

# load the images into memory start with NORMAL
#normalImages, normalLabels = load_images(normalPath)

# load the images into memory now do PNEUMONIA
#pnImages, pnLabels = load_images(pnPath)

#xTrain = np.append(normalImages, pnImages, axis=0)
#yTrain = np.append(normalLabels, pnLabels)




#
# Show 14 images at random, see if they are loading correctly (Don't need this, just confirming
# that we have loaded the images correctly)
#
'''
fig, axes = plt.subplots(ncols=7, nrows=2, figsize=(16, 4))

indices = np.random.choice(len(xTrain), 14)
counter = 0

for i in range(2):
    for j in range(7):
        axes[i, j].set_title(yTrain[indices[counter]])
        axes[i, j].imshow(xTrain[indices[counter]], cmap='gray')
        axes[i, j].get_xaxis().set_visible(False)
        axes[i, j].get_yaxis().set_visible(False)
        counter += 1
plt.show()
'''


#
# IF THERE ISN'T A '.PICKLE' FILE THEN UNCOMMENT THESE LINES!!!!!
#

# Read in the test images now
#norm_images_test, norm_labels_test = load_images(normalPathT)
#pneu_images_test, pneu_labels_test = load_images(pnPathT)

# Save the test data into np arrays
#X_test = np.append(norm_images_test, pneu_images_test, axis=0)
#y_test = np.append(norm_labels_test, pneu_labels_test)

# Use this to save variables (Converting object data into byte data)
# (serializing and deserializing a Python object structure)
#with open('pneumonia_data.pickle', 'wb') as f:
#    pickle.dump((xTrain, X_test, yTrain, y_test), f)# Use this to load variables


#
# IF THERE ISN'T A '.PICKLE' FILE THEN COMMENT THESE LINES!!!!!
#

with open('pneumonia_data.pickle', 'rb') as f:
    (xTrain, X_test, yTrain, y_test) = pickle.load(f)


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

counter = 2.4
counterz = 0.4
counterw = 0.8
counterh = 0.6
file = open(fileOutput, 'a')

for x in range(16):
    counter += 2
    counterz = 0

    for z in range(3):
        counterz += 0.4
        counterw = 0

        for w in range(3):
            counterw += 0.4
            counterh = 0

            for h in range(3):
                counterh += 0.4
                # Create more training samples by changing the photos. (zoom in, rotate image, etc)
                datagen = ImageDataGenerator(
                    rotation_range=counter,
                    zoom_range=counterz,
                    width_shift_range=counterw,
                    height_shift_range=counterh)

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
                cnn = Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
                             padding='same')(input1)
                cnn = Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
                             padding='same')(cnn)
                cnn = MaxPool2D((2, 2))(cnn)

                cnn = Conv2D(16, (2, 2), activation='relu', strides=(1, 1),
                             padding='same')(cnn)
                cnn = Conv2D(32, (2, 2), activation='relu', strides=(1, 1),
                             padding='same')(cnn)
                cnn = MaxPool2D((2, 2))(cnn)

                #
                #  Neural network look at documentation for 'Flatten', 'Dense'
                # https://stackoverflow.com/questions/43237124/what-is-the-role-of-flatten-in-keras
                # https://keras.io/api/layers/core_layers/dense/
                #
                cnn = Flatten()(cnn)
                cnn = Dense(100, activation='relu')(cnn)
                cnn = Dense(50, activation='relu')(cnn)
                output1 = Dense(3, activation='softmax')(cnn)

                #
                # Model groups layers into an object with training and inference features.
                # https://keras.io/api/models/model/#model-class
                #
                model = Model(inputs=input1, outputs=output1)

                # https://keras.io/api/models/model_training_apis/#compile-method
                model.compile(loss='categorical_crossentropy',
                              optimizer='adam', metrics=['acc'])

                # https://www.geeksforgeeks.org/keras-fit-and-keras-fit_generator/
                history = model.fit_generator(train_gen, epochs=epochs,
                                              validation_data=(X_test, y_test_one_hot))

                # print("Accuracy: ", history.history['acc'], "\nValue Accuracy: ", history.history['val_acc'], "\n\n")

                printVar = "rotation_range: " + str(counter)
                printZoomRange = 'zoom_range: ' + str(counterz)
                printWidth_shift_range = 'width_shift_range: ' + str(counterw)
                printHeight_shift_range = 'height_shift_range: ' + str(counterh)

                file.write(printVar)
                file.write('\n')

                file.write(printZoomRange)
                file.write('\n')

                file.write(printWidth_shift_range)
                file.write('\n')

                file.write(printHeight_shift_range)
                file.write('\n')

                file.write("Accuracy: ")
                averageAcc = 0
                averageVal = 0

                for e in history.history['acc']:
                    file.write(str(e))
                    file.write(' ')
                    averageAcc += e
                averageAcc = averageAcc / len(history.history['acc'])
                file.write('\nAverage Acc: ')
                file.write(str(averageAcc))

                file.write("\nValue Accuracy: ")
                for e in history.history['val_acc']:
                    file.write(str(e))
                    file.write(' ')
                    averageVal += e
                averageVal = averageVal / len(history.history['val_acc'])
                file.write('\nAverage Val: ')
                file.write(str(averageVal))
                file.write("\n")

                # output = "Accuracy: " + history.history['acc'] + "\nValue Accuracy: " + history.history['val_acc'] + "\n"
                # acc_hist = listToString(history.history['acc'])
                # val_hist = listToString(history.history['val_acc'])

                # output = "Accuracy: " + acc_hist + "\nValue Accuracy: " + val_hist
                # file.write(output)

                '''
                # Plot the first values
                plt.figure(figsize=(8, 6))
                plt.title('Accuracy scores')
                plt.plot(history.history['acc'])
                plt.plot(history.history['val_acc'])
                plt.legend(['acc', 'val_acc'])
                plt.show()

                # Plot the second values
                plt.figure(figsize=(8, 6))
                plt.title('Loss value')
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.legend(['loss', 'val_loss'])
                plt.show()
                '''
                # Get the predictions from the neural network
                predictions = model.predict(X_test)
                predictions = one_hot_encoder.inverse_transform(predictions)

                # Make the confusion matrix with the predictions vs the actual values
                cm = confusion_matrix(y_test, predictions)
                file.write('Confusion Matrix')

                for e in cm:
                    file.write(str(e))
                file.write('\n')

                tp = cm[0, 0]
                tn = cm[1, 0] + cm[1, 1] + cm[2, 0] + cm[2, 1]
                fp = cm[0, 1] + cm[0, 2]
                fn = cm[1, 2] + cm[2, 2]

                file.write('TP: ')
                file.write(str(tp))
                file.write('\n')

                file.write('TN: ')
                file.write(str(tn))
                file.write('\n')

                file.write('FP: ')
                file.write(str(fp))
                file.write('\n')

                file.write('FN: ')
                file.write(str(fn))

                file.write('\n\n')
                '''
                # Visualize the confusion matrix
                classnames = ['bacteria', 'normal', 'virus']
                plt.figure(figsize=(8, 8))
                plt.title('Confusion matrix')
                sns.heatmap(cm, cbar=False, xticklabels=classnames, yticklabels=classnames, fmt='d', annot=True, cmap=plt.cm.Blues)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.show()
                '''


file.close()




