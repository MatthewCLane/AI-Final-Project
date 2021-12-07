# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\TestInterface.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import os, sys
import numpy as np
import cv2.cv2
from keras.models import load_model
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(679, 561)
        self.ProgramFrame = QtWidgets.QFrame(Dialog)
        self.ProgramFrame.setGeometry(QtCore.QRect(-1, -1, 681, 561))
        self.ProgramFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.ProgramFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.ProgramFrame.setObjectName("ProgramFrame")
        self.groupBox = QtWidgets.QGroupBox(self.ProgramFrame)
        self.groupBox.setGeometry(QtCore.QRect(250, 480, 181, 71))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setObjectName("pushButton")
        self.pushButton_2.clicked.connect(self.getfile)
        self.gridLayout_2.addWidget(self.pushButton_2, 0, 0, 1, 1)
        self.ImageFrame = QtWidgets.QFrame(self.ProgramFrame)
        self.ImageFrame.setGeometry(QtCore.QRect(29, 29, 400, 400))
        self.ImageFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.ImageFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.ImageFrame.setObjectName("ImageFrame")
        self.XrayImage = QtWidgets.QGraphicsView(self.ImageFrame)
        self.XrayImage.setGeometry(QtCore.QRect(10, 10, 400, 400))
        self.XrayImage.setObjectName("XrayImage")
        #X-Ray Image Panel Initial Image
        #Create the scene for the X-Ray Image
        self.scene = QtWidgets.QGraphicsScene(self.XrayImage)
        #We want the the inital bacground to be a light gray to indicate that it's off.
        self.scene.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.lightGray, QtCore.Qt.SolidPattern))
        #Create a pixmap, scale it, and add it to the scene.
        #By Stillwaterising - Own work, CC0, https://commons.wikimedia.org/w/index.php?curid=9686540
        self.pixmap = QPixmap("Chest_Xray_PA_3-8-2010.png")
        #Scale image to fit within panel
        self.pixmap = self.pixmap.scaled(397, 397, QtCore.Qt.KeepAspectRatio)
        self.item = QtWidgets.QGraphicsPixmapItem(self.pixmap)
        # Since this is the initial run, let's gray out the image, like the X-Ray film viewer isn't turned on
        self.effect = QtWidgets.QGraphicsColorizeEffect(self.scene)
        self.effect.setStrength(0.75)
        self.effect.setColor(QtGui.QColor('white'))
        self.item.setGraphicsEffect(self.effect)
        #Finally add item to scene.
        self.scene.addItem(self.item)
        # Center the image in the scene
        self.item.setTransformOriginPoint(self.item.boundingRect().center())
        self.XrayImage.setScene(self.scene)
        self.XrayImage.setCacheMode(QtWidgets.QGraphicsView.CacheBackground)

        self.ResultFrame = QtWidgets.QFrame(self.ProgramFrame)
        self.ResultFrame.setGeometry(QtCore.QRect(430, 30, 241, 421))
        self.ResultFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.ResultFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.ResultFrame.setObjectName("ResultFrame")
        self.ResultText = QtWidgets.QTextBrowser(self.ResultFrame)
        self.ResultText.setGeometry(QtCore.QRect(10, 60, 231, 341))
        self.ResultText.setObjectName("ResultText")
        self.ResultText.setText(
            "<p>" + "<strong>" + "Predicted: " + "</strong>" + "<br>" + "<p>" + "<strong>" + " Certainty: " + "</strong>" )
        self.label = QtWidgets.QLabel(self.ResultFrame)
        self.label.setGeometry(QtCore.QRect(100, 20, 55, 16))
        self.label.setObjectName("label")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.groupBox.setTitle(_translate("Dialog", ""))
        self.pushButton_2.setText(_translate("Dialog", "Load X-ray"))
        self.pushButton_2.setFont
        self.label.setText(_translate("Dialog", "Result"))

    def getfile(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(None, 'Open file', os.getcwd(), 'Images(*.png *.gif *.jpg, *.jpeg)')
        #Since this is the first/subsequent runs, set background to white.
        self.scene.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.white, QtCore.Qt.SolidPattern))
        #Paint the pixmap white and re-add it to the scene/panel
        self.pixmap.fill(QtCore.Qt.white)
        self.item = QtWidgets.QGraphicsPixmapItem(self.pixmap)
        # Finally add item to scene.
        self.scene.addItem(self.item)
        #Reset the result fields
        self.ResultText.setText(
            "<p>" + "<strong>" + "Predicted: " + "</strong>" + "<br>" + "<p>" + "<strong>" + " Certainty: " + "</strong>")
        #Now that we have a blank slate, request image from the user.
        self.pixmap = QPixmap(fname[0])
        #Check that a file was loaded, load blank if not.
        if fname[0] != '':
            self.pixmap = self.pixmap.scaled(397, 397, QtCore.Qt.KeepAspectRatio)
            self.item = QtWidgets.QGraphicsPixmapItem(self.pixmap)
            # Finally add item to scene.
            self.scene.addItem(self.item)
        # Center the image in the scene
        self.item.setTransformOriginPoint(self.item.boundingRect().center())
        self.XrayImage.setScene(self.scene)
        self.XrayImage.setCacheMode(QtWidgets.QGraphicsView.CacheBackground)
        if fname[0] != '':
            self.runCNN(fname[0])

    def runCNN(self, fname):
        #
        # This is where we will send the image through the CNN.
        # All we have to do here is do the same thing to the images that
        # We did in the OG project. Then load in the model, and use the predict function.
        #

        # Define an array
        imgArray = []

        # Read the image in using opencv
        modelImg = cv2.cv2.imread(fname)

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
        model = load_model('E9085.09L1N150L2N610_77.h5')

        # Send the image into the CNN. Will return an array of 3 numpyFloats.
        # Highest number is what the AI predicts. (Links directly up with the options)
        prediction = model.predict(imgArray)
        #print(prediction)

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

        # Now that we know what the highest numbers index is, we can use that in our options array to get
        # The text that our AI predicts. (Creates the Label)
        #   options[index]
        #And the percentage
        #   prediction[0][index]

        # Add these to the GUi
        self.displayresult(options[index], prediction[0][index])

    def displayresult(self, result, percentage):
        self.ResultText.setText("<p>" + "<strong>" + "Predicted: " + "</strong>" + result + "<br>" + "<p>" + "<strong>" + " Certainty: " + "</strong>" + str(round((percentage*100), 2)) + '%')

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet("QTextBrowser { font-size: 10pt}" "QLabel, QPushButton {font-size: 10pt; font-weight: bold}")
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())