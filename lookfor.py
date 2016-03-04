## Counts specified words in training files
## Reports their average occurance, standard deviation, likelihood of being absent within each class
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
    data[type]['numsocks'] = []
    data[type]['numdests'] = []
    data[type]['numsets'] = []
    data[type]['numdels'] = []
    data[type]['numtargs'] = []
    data[type]['numgets'] = []

    for filename in onlynames:

        if filename.split('.')[-2] == type:

            reading = open('train/' + filename)
            text = reading.read()

            data[type]['numsocks'].append(text.count("socket"))
            data[type]['numdests'].append(text.count("destroy_window"))
            data[type]['numsets'].append(text.count("set_value"))
            data[type]['numdels'].append(text.count("delete_value"))
            data[type]['numtargs'].append(text.count("targetpid"))
            data[type]['numgets'].append(text.count("get_file_attributes"))


for set in data[type]:
    print '\n' + set + '\n'
    for type in classes:
        list = data[type][set]
        zeroes = list.count(0)
        zratio  = float(zeroes)/len(list)
        array = np.array(data[type][set])
        print "TYPE: %s" % type
        print "AVG: %.2f, STD: %.2f, ZERO: %.2f" % (np.mean(array), np.std(array), zratio)
        # print data[type][set]
