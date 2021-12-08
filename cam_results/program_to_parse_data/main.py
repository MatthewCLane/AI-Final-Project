'''
November 13, 2021
This is part of the week2 assignments where I have to determine which
iteration of the AI had the best accuracy according to the results
that Brandon recorded from one of his runs

The file I am referring to is FinalSamples.txt

This is from the experimentation branch

November 15, 2021
I duplicate this project to search for Brandon's new entries for test set average.
Brandon's additions were done in a file called FinalSamples_Ordered_1.txt
'''

# import statements
import re

# functions
def printListToFile(L,fileArg):
    for element in L:
        print(element, end="", file=fileArg)
    print(file=fileArg)


# lists
listOfAverageAccLines = []                          # to store list of lines containing avg acc to find greatest one
listOfAverageAcc = []                               # to hold actual average accuracies parsed from lines

with open("FinalSamples_Ordered_1.txt") as sampleFile:  # this technique auto closes the file when done (with...as...)
    line = sampleFile.readline()                    # read next line and then loop while there are more lines
    while line:
        line = sampleFile.readline()
        # if "Average Acc:" in line:                  # finding lines with average accuracy
        if "Test Set Average:" in line:             # finding lines with average test set accuracy
            line = line.replace('\n', '')           # removing new line character
            listOfAverageAccLines.append(line)

#for s in listOfAverageAccLines:
    #print(s)

for numbers in listOfAverageAccLines:               # to parse through list of lines and grab the numbers only
    tempList = []
    regPatt = re.compile('[0-9]*.[0-9]+')           # regex for floating number

    tempList = numbers.split(' ')                   # splitting based on space
    if regPatt.match( tempList[len(tempList)-1] ):  # checking if end of line contains float value
        listOfAverageAcc.append( float(tempList[len(tempList)-1]) )     # appending avg acc cast as float to list

listOfAverageAcc = sorted(listOfAverageAcc, reverse=True)               # sorting list greatest to least

for s in listOfAverageAcc:
    print(s)

writeFile = open("FinalSamples_Ordered_TestSetAcc.txt", "w")
for flo in listOfAverageAcc:                                # looping through list of averages
    tempFloat_asString = "Test Set Average: " + str(flo)    # casting float from list as string
    with open("FinalSamples_Ordered_1.txt") as sampleFile:  # looping through file
        line = sampleFile.readline()                        # temp variable for lines of file
        tempList_forGrabbingDataChunks = []                 # this list will contain lines of the chunks of data
        while line:
            line = sampleFile.readline()
            if len(line) > 5:                           # if line is not empty, add to list
                tempList_forGrabbingDataChunks.append(line)
            if len(line) < 5:                           # if line is empty, print everything out and reset list
                for dataChunk in tempList_forGrabbingDataChunks:
                    if tempFloat_asString in dataChunk:
                        printListToFile(tempList_forGrabbingDataChunks, writeFile)
                tempList_forGrabbingDataChunks.clear()  # clearing the list

