## Counts specified words in training files.
## Reports their average occurance, standard deviation, likelihood of absence,
## and average ignoring absences within each class - all divided by a fraction of the total line
## count so as to lessen the file length's effect on the results.
## Assumes to be in a directory along with the training set directory labeled "train".

import numpy as np
from os import listdir
from os.path import isfile, join

classes = ["Agent", "AutoRun", "FraudLoad", "FraudPack", "Hupigon", "Krap",
           "Lipler", "Magania", "None", "Poison", "Swizzor", "Tdss",
           "VB", "Virut", "Zbot"]
filenames = [f for f in listdir('train') if isfile(join('train', f))]
words = ['socket', 'destroy_window', 'set_value', 'delete_value', 'targetpid', 'get_file_attributes',
         'FILE_ANY_ACCESS', 'FILE_ATTRIBUTE_NORMAL', 'SECURITY_ANONYMOUS', 'import', 'export']
data = {}

for type in classes:

    data[type] = {}
    for word in words:
        data[type][word] = []

    for filename in filenames:

        if filename.split('.')[-2] == type:

            reading = open('train/' + filename)
            text = reading.read()

            length = len(text.split("\n"))/1000.0
            length = 1

            for word in words:
                data[type][word].append(text.count(word)/length)

for set in data['Agent']:
    if set != 'length':
        print '\n **' + set + '** \n'
        for type in classes:
            list = data[type][set]
            zeroes = list.count(0)
            zratio  = float(zeroes)/len(list)
            if zeroes < len(list):
                nozavg = float(sum(list)) / (len(list) - zeroes)
            else:
                nozavg = -1
            array = np.array(list)
            print "TYPE: %s" % type
            print "AVG: %.2f, STD: %.2f, AVG-0: %.2f ZERO: %.2f" % (np.mean(array), np.std(array), nozavg, zratio)
            # print data[type][set]
