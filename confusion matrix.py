'''
This program was run at the start of our project to see what the confusion matrix values were.

By: Brandon Strong

'''

# End the program when there is a keyboard interupt
try:

    # Loop that always runs
    while True:

        # Ask the user what the variables of the confusion matrix are (Values from 1-9)
        var1 = input('var1: ')
        var2 = input('var2: ')
        var3 = input('var3: ')
        var4 = input('var4: ')
        var5 = input('var5: ')
        var6 = input('var6: ')
        var7 = input('var7: ')
        var8 = input('var8: ')
        var9 = input('var9: ')

        # Convert those numbers into an integer so we can do math on them
        var1 = int(var1)
        var2 = int(var2)
        var3 = int(var3)
        var4 = int(var4)
        var5 = int(var5)
        var6 = int(var6)
        var7 = int(var7)
        var8 = int(var8)
        var9 = int(var9)

        #
        # Note there are 624 images that we are using in the test set. A simple change would be to add up all of
        # the vars and use that in place of 624. That way this would work for any 3x3 confusion matrix
        #

        # Calculate all of the desired information that we want to know
        accuracy = (((var1 + var5) + var9) / 624) * 100
        sensitivity = (var1 / (var1 + var2 + var3)) * 100
        specificity = ((var5 + var6 + var8 + var9) / (var4 + var5 + var6 + var7 + var8 + var9)) * 100
        ppv = (var1 / (var1 + var4 + var7)) * 100
        npv = ((var5 + var6 + var8 + var9) / (var2 + var3 + var5 + var6 + var8 + var9)) * 100
        accuracyBinary = (((var1 + var5) + var9 + var3 + var7) / 624) * 100

        # Print the information to the user
        print('Accuracy: ' + str(accuracy) + '%')
        print('Sensitivity: ' + str(sensitivity) + '%')
        print('Specificity: ' + str(specificity) + '%')
        print('Positive Predictive Value: ' + str(ppv) + '%')
        print('Negative Predictive Value: ' + str(npv) + '%')
        print('Binary Accuracy: ' + str(accuracyBinary) + '%')
        print('\n')

except KeyboardInterrupt:
    pass

