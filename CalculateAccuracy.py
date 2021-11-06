


fileOutput = 'C:\\Users\\asus\\Desktop\\samples.txt'
newFile = 'C:\\Users\\asus\\Desktop\\FinalSamples.txt'

file = open(fileOutput, 'r')
newFileAppend = open(newFile, 'a')

Lines = file.readlines()



for line in Lines:
    total = 0
    accuracy = 0
    flag = False
    stringBuilder = ''

    if 'TP' in line:
        for i in line:
            if i == ' ' and flag == False:
                flag = True
            if flag == True:
                if i != '\n':
                    stringBuilder = stringBuilder + i
        tp = int(stringBuilder)
    elif 'TN' in line:
        for i in line:
            if i == ' ' and flag == False:
                flag = True
            if flag == True:
                if i != '\n':
                    stringBuilder = stringBuilder + i
        tn = int(stringBuilder)
    elif 'FP' in line:
        for i in line:
            if i == ' ' and flag == False:
                flag = True
            if flag == True:
                if i != '\n':
                    stringBuilder = stringBuilder + i
        fp = int(stringBuilder)

    elif 'FN' in line:
        for i in line:
            if i == ' ' and flag == False:
                flag = True
            if flag == True:
                if i != '\n':
                    stringBuilder = stringBuilder + i

        fn = int(stringBuilder)
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total
        newFileAppend.write('Accuracy: ')
        newFileAppend.write(str(accuracy))
        newFileAppend.write('\n')

        tp = 0
        tn = 0
        fp = 0
        fn = 0
    else:
        newFileAppend.write(line)
file.close()
newFileAppend.close()