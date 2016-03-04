## Counts specified words in training files
## Reports their average occurance, standard deviation, likelihood of absence,
## and average ignoring absences within each class - all divided a fraction of the total line
## count so as to lessen file length's effect on the results
## Assumes to be in a directory along with the training set directory labeled "train"

import numpy as np
from os import listdir
from os.path import isfile, join

classes = ["Agent", "AutoRun", "FraudLoad", "FraudPack", "Hupigon", "Krap",
           "Lipler", "Magania", "None", "Poison", "Swizzor", "Tdss",
           "VB", "Virut", "Zbot"]

onlynames = [f for f in listdir('train') if isfile(join('train', f))]

data = {}

for type in classes:

    data[type] = {}
    data[type]['socket'] = []
    data[type]['destroy_window'] = []
    data[type]['set_value'] = []
    data[type]['delete_value'] = []
    data[type]['targetpid'] = []
    data[type]['get_file_attributes'] = []
    data[type]['FILE_ANY_ACCESS'] = []
    data[type]['FILE_ATTRIBUTE_NORMAL'] = []
    data[type]['SECURITY_ANONYMOUS'] = []


    for filename in onlynames:

        if filename.split('.')[-2] == type:

            reading = open('train/' + filename)
            text = reading.read()

            length = len(text.split("\n"))/1000.0
            length = 1

            data[type]['socket'].append(text.count("socket")/length)
            data[type]['destroy_window'].append(text.count("destroy_window")/length)
            data[type]['set_value'].append(text.count("set_value")/length)
            data[type]['delete_value'].append(text.count("delete_value")/length)
            data[type]['targetpid'].append(text.count("targetpid")/length)
            data[type]['get_file_attributes'].append(text.count("get_file_attributes")/length)
            data[type]['FILE_ANY_ACCESS'].append(text.count("FILE_ANY_ACCESS")/length)
            data[type]['FILE_ATTRIBUTE_NORMAL'].append(text.count("FILE_ATTRIBUTE_NORMAL")/length)
            data[type]['SECURITY_ANONYMOUS'].append(text.count("SECURITY_ANONYMOUS")/length)


for set in data['Agent']:
    if set != 'length':
        print '\n' + set + '\n'
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
