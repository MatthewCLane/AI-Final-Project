from tkinter import *
from tkinter import filedialog as fd
import numpy as np
import cv2.cv2
from PIL import ImageTk, Image
from keras.models import load_model

# Create the GUI window
root = Tk()

# Name the GUI window
root.title('Testing Tkinter')

# Have an image as the icon
root.iconbitmap('testIcon.ico') # needs to be an ico file

# Setting the GUI size
root.geometry('1280x720')

#Setting the minimum size the GUI can be
root.minsize(1280, 720)

# Centering the columns in the GUI
root.columnconfigure(0, weight=1)
#root.rowconfigure(0, weight=1)

# Creating the button to quit the program. (Not displayed... Not sure if it would be useful)
button_quit = Button(root, text='Quit Program', command=root.quit)

def testing():

    # Defining the image as a global variable
    global image

    # Has the user select a file. Gets the path to that file
    fileLocation = fd.askopenfilename()

    # Opens the image
    image = Image.open(fileLocation)

    # Resizes the image to be smaller or bigger
    resizedImg = image.resize((700, 525))

    # Function to make the photo ready to be added in the GUI
    image = ImageTk.PhotoImage(resizedImg)

    # Add the photo to a label
    labelImg = Label(root, image=image)

    # Display the label in the GUI
    labelImg.grid(row=1, column=0)

    #
    # This is where we will send the image through the CNN.
    # All we have to do here is do the same thing to the images that
    # We did in the OG project. Then load in the model, and use the predict function.
    #

    # Define an array
    imgArray = []

    # Read the image in using opencv
    modelImg = cv2.cv2.imread(fileLocation)

    # Resize the image to be 200x200
    modelImg = cv2.cv2.resize(modelImg, dsize=(200, 200))

    # Convert the image into gray scale
    modelImg = cv2.cvtColor(modelImg, cv2.COLOR_BGR2GRAY)

    # Append the image to the array
    imgArray.append(modelImg)

    # Convert the array to a numpy array
    imgArray = np.array(imgArray)

    # These are the options that the AI will choose from
    options = ['Bacteria Pneumonia', 'Normal', 'Virus Pneumonia']

    # Load in the model
    model = load_model('my_model.h5')

    # Send the image into the CNN. Will return an array of 3 numpyFloats.
    # Highest number is what the AI predicts. (Links directly up with the options)
    prediction = model.predict(imgArray)

    #
    # Find the biggest number in prediction, and get that numbers index
    #
    index = 0
    iteration = 0
    biggest = np.float32(0)
    for i in prediction:
        for l in i:
            if l > biggest:
                biggest = l
                index = iteration
            iteration += 1

    # Try to fix bug. Didn't work lol
    predictionLabel = Label(root, text='')
    predictionLabel.grid_forget()
    predictionLabel.config(font=60)
    predictionLabel.grid(row=2, column=0)

    # Now that we know what the highest numbers index is, we can use that in our options array to get
    # The text that our AI predicts. (Creates the Label)
    predictionLabel = Label(root, text=options[index])

    # Change the size of the font
    predictionLabel.config(font=60)

    # Add the label to the GUI
    predictionLabel.grid(row=2, column=0)


# Create the button to open a picture file
myButton = Button(root, text='Open a file', padx=25, pady=25, command=testing)

# Display the button on the GUI
myButton.grid(row=0, column=0)

# Run the whole GUI
root.mainloop()