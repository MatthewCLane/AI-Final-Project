'''
This is the file that we ran to get the pytorch trinary results.

Edited By: Brandon Strong
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
import timm
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch import nn
from torchvision import transforms as T, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F


# This checks to see if there is a GPU that can be used to run the AI
print('Is the GPU available: ', torch.cuda.is_available())

# This is to see if you can use your gpu if you have one
# a=torch.cuda.FloatTensor()
# print(a)

# Get the accuracy and loss over each epoch stored in these variables
accuracyTrainArray = []
accuracyTestArray = []
lossTrainArray = []
lossTestArray = []
testSetLogits = []


class CFG:
    # Settings for the AI
    epochs = 120
    lr = 0.001
    batch_size = 16

    # Model we are importing from timm
    model_name = 'tf_efficientnet_b4_ns'
    img_size = 224

    # Going to be use for loading dataset 
    DATA_DIR = 'C:\\Users\\tec05\\Desktop\\chest_xray'
    TEST = 'test'
    TRAIN = 'train'
    VAL = 'val'


# Prints the device that the user is going to use to run the AI
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Which device we are on : {}".format(device))


# Resizes the images and adds rotation to the images (For the training)
train_transform = T.Compose([

    T.Resize(size=(CFG.img_size, CFG.img_size)),
    T.RandomRotation(degrees=(-20, +20)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Resize the images
valid_transform = T.Compose([

    T.Resize(size=(CFG.img_size, CFG.img_size)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Resize the images
test_transform = T.Compose([

    T.Resize(size=(CFG.img_size, CFG.img_size)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Get the path to the dataset
train_path = os.path.join(CFG.DATA_DIR, CFG.TRAIN)
valid_path = os.path.join(CFG.DATA_DIR, CFG.VAL)
test_path = os.path.join(CFG.DATA_DIR, CFG.TEST)

# Create arrays that you can access each photo
trainset = datasets.ImageFolder(train_path, transform=train_transform)
validset = datasets.ImageFolder(valid_path, transform=valid_transform)
testset = datasets.ImageFolder(test_path, transform=test_transform)

# Create arrays to store the labels and images to the test set
testSetLabels = []
testSetImages = []

# Store the class names in an array
class_names = ['Bacteria Pneumonia', 'Normal', 'Virus Pneumonia']

# Load all of the data into batches
trainloader = DataLoader(trainset, batch_size=CFG.batch_size, shuffle=True)
validloader = DataLoader(validset, batch_size=CFG.batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=CFG.batch_size, shuffle=True)

# This one is for testing the raw images. It's not shuffled so we can verify the prediction results
testSetCM = DataLoader(testset, shuffle=False)

print('Loading the test set images into an array')
for i, _ in tqdm(testSetCM):
    testSetImages.append(i.to(device))

print('Loading test set labels into an array')
# Get the labels stored into an array
for dump, i in tqdm(testset):
    testSetLabels.append(i)

# We create the model here
model = timm.create_model(CFG.model_name, pretrained=True)

# Set parameters to False
for param in model.parameters():
    param.requires_grad = False

# This is where we set the layers Neurons to. start with 1792 which is the width * 8.
# Then end with 3 classifications
model.classifier = nn.Sequential(

    nn.Linear(in_features=1792, out_features=625),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(in_features=625, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=3)

)

# Use the CPU or the GPU to compute the model
model.to(device)


# Function to calculate accuracy
def accuracy(y_pred, y_true):
    # Get the softmax of the prediction value
    y_pred = F.softmax(y_pred, dim=1)

    # Get the the prediction that has the highest number
    top_p, top_class = y_pred.topk(1, dim=1)

    equals = top_class == y_true.view(*top_class.shape)

    # Return the mean of the accuracy
    return torch.mean(equals.type(torch.FloatTensor))


class PneumoniaTrainer():

    def __init__(self, criterion=None, optimizer=None, schedular=None):

        self.criterion = criterion
        self.optimizer = optimizer
        self.schedular = schedular

    def train_batch_loop(self, model, trainloader):

        train_loss = 0.0
        train_acc = 0.0

        # Going through each of the images in the train set
        for images, labels in tqdm(trainloader):
            # Use the GPU or CPU to calculate
            images = images.to(device)
            labels = labels.to(device)

            # Get the logits from the model
            logits = model(images)

            # Get the loss from the labels
            loss = self.criterion(logits, labels)

            # Set the optimizer to zero gradient
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Keep track of all the losses for the AI. Add over and over
            train_loss += loss.item()

            # Keep track of all the accuracy for the AI. Add over and over
            train_acc += accuracy(logits, labels)

        return train_loss / len(trainloader), train_acc / len(trainloader)

    # Doing the same thing as the function above, just for the validation
    def valid_batch_loop(self, model, validloader):

        valid_loss = 0.0
        valid_acc = 0.0

        for images, labels in tqdm(validloader):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            testSetLogits.append(logits)
            loss = self.criterion(logits, labels)

            valid_loss += loss.item()

            valid_acc += accuracy(logits, labels)

        return valid_loss / len(validloader), valid_acc / len(validloader)

    # This is where we will actually train the model
    def fit(self, model, trainloader, validloader, epochs):

        valid_min_loss = np.Inf

        # Go for as many epochs there is specified
        for i in range(epochs):

            # Train the model
            model.train()

            # Train off of the batch
            avg_train_loss, avg_train_acc = self.train_batch_loop(model, trainloader)

            # Turn off certain tensors to do the validation set
            model.eval()
            avg_valid_loss, avg_valid_acc = self.valid_batch_loop(model, validloader)

            # display a message if the loss is less than the previous minimum. Save the file
            # and update the new lowest value.
            if avg_valid_loss <= valid_min_loss:
                # print("Valid_loss decreased {} --> {}".format(valid_min_loss, avg_valid_loss))
                torch.save(model.state_dict(), 'ColabPneumoniaModel.pt')
                valid_min_loss = avg_valid_loss

            # Append the accuracy and loss for the train and the test set
            accuracyTrainArray.append(avg_train_acc)
            accuracyTestArray.append(avg_valid_acc)
            lossTrainArray.append(avg_train_loss)
            lossTestArray.append(avg_valid_loss)

            # Print the restults to the user
            print("Epoch : {} Train Loss : {:.6f} Train Acc : {:.6f}".format(i + 1, avg_train_loss, avg_train_acc))
            print("Epoch : {} Valid Loss : {:.6f} Valid Acc : {:.6f}".format(i + 1, avg_valid_loss, avg_valid_acc))


"""# Training Model """

# Cross Entropy for the NN
criterion = nn.CrossEntropyLoss()

# Have the optimizer be adam for the NN
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)

# Create the class with the optimizer and cross entropy
trainer = PneumoniaTrainer(criterion, optimizer)

# Start training the model, pass in the train set and the test set
print('Training started # of epochs: ', CFG.epochs)
trainer.fit(model, trainloader, testloader, epochs=CFG.epochs)

#
# Start plotting
#

# Load the model from the one that we have trained
model.load_state_dict(torch.load('ColabPneumoniaModel.pt'))

# Turn off certain tensors to train the test set
model.eval()

avg_test_loss, avg_test_acc = trainer.valid_batch_loop(model, validloader)

# Print the test loss and the test accuracy to the users
print("Validation Set Loss : {}".format(avg_test_loss))
print("Validation Set Accuracy : {}".format(avg_test_acc))

# Get the 324th image from the test set
image, label = testset[324]

# Make sure that the label is the same as what it should be
# print('Should be normal: ', class_names[label])

# Get the 240th image from the test set
image, label = testset[240]

# Plot the accuracy of the test set vs the training set
plt.figure(figsize=(8, 6))
plt.title('Accuracy Scores')
plt.plot(accuracyTrainArray)
plt.plot(accuracyTestArray)
plt.legend(['Acc', 'Val_Acc'])
plt.savefig('accuracyPytorch.png', bbox_inches='tight')

# An array to store the predictions of the test set
predictions = []

# Getting the predictions from the model for the test set
print('Running the test images into the model')
for i in tqdm(testSetImages):
    prediction = model(i)

#print(prediction)

    # Choosing the highest score, and appending the index to the predictions array
    predictions.append(np.argmax(prediction.cpu().detach().numpy()))
print('Making the plots')

# Calculate the confusion matrix with the correct labels and the predictions
cm = confusion_matrix(testSetLabels, predictions)

# Plot the confusion matrix for the test set
plt.figure(figsize=(8, 8))
plt.title('Confusion Matrix')
sns.heatmap(cm, cbar=False, xticklabels=class_names, yticklabels=class_names, fmt='d', annot=True, cmap=plt.cm.Blues)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusionMatrixPytorch.png', bbox_inches='tight')

# Plot the loss of the training set vs the test set
plt.figure(figsize=(8, 6))
plt.title('Loss Value')
plt.plot(lossTrainArray)
plt.plot(lossTestArray)
plt.legend(['Loss', 'Val_Loss'])
plt.savefig('lossPytorch.png', bbox_inches='tight')
