# Example Feature Extraction from XML Files
# We count the number of specific system calls made by the programs, and use
# these as our features.

# This code requires that the unzipped training set is in a folder called "train". 

import os
from collections import Counter
from datetime import datetime
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
import cPickle as pickle
import sys
from scikits.statsmodels.distributions import ECDF # installed from http://scikits.appspot.com/statsmodels

import util

TRAIN_DIR = "train"

call_list = pickle.load(open('tag_list.p', 'r'))
call_set = set(call_list)


#features = ['sleep', 'dump_line']
features = call_list + map(lambda s : s + ' indicator', call_list) + map(str, range(4000))\
           + ['has_socket', 'has_import', 'has_export', 'bytes_sent', 'bytes_received', 'any_sent', 'any_received', 'totaltime']
#features = call_list + ['bytes_sent', 'bytes_received', 'any_sent', 'any_received', 'totaltime']



#def add_to_set(tree):
#    for el in tree.iter():
#        call = el.tag
#        call_set.add(call)

def create_data_matrix(num=None):
    X = None
    classes = []
    ids = [] 
    i = -1

    train_files = os.listdir('train')
    test_files = os.listdir('test')

    for datafile in train_files + test_files:
        if datafile == '.DS_Store':
            continue

        i += 1
        if i % 100 == 0:
            print i
            sys.stdout.flush()
        if num is not None and i >= num:
            break

        # extract id and true class (if available) from filename
        id_str, clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))

        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)

        # parse file as an xml document
        direc = 'train' if clazz != 'X' else 'test'
        tree = ET.parse(os.path.join(direc, datafile))

        this_row_dict = call_feats(tree)
        this_row_dict['has_socket'] = int('socket' in open(os.path.join(direc, datafile), 'r').read())
        this_row_dict['has_import'] = int('import' in open(os.path.join(direc, datafile), 'r').read())
        this_row_dict['has_export'] = int('export' in open(os.path.join(direc, datafile), 'r').read())
        this_row = np.array([(this_row_dict[feature] if feature in this_row_dict else 0) for feature in features])
        #this_row = call_feats(tree)
        if X is None:
            X = this_row
        else:
            X = np.vstack((X, this_row))

    X = X.T
    for i in range(len(X)):
        if X[i].any() and len(set(X[i])) > 2:
            X = np.vstack((X, ECDF(X[i])(X[i])))
            features.append(features[i] + ' ECDF')
    X = X.T

    return X, np.array(classes), ids, features

def call_feats(tree):
    #good_calls = ['sleep', 'dump_line']
    #good_calls = call_list + ['bytes_sent', 'bytes_received', 'totaltime']

    call_counter = {}
    call_counter['bytes_sent'] = 0
    call_counter['bytes_received'] = 0
    call_counter['any_sent'] = 0
    call_counter['any_received'] = 0
    call_counter['totaltime'] = 0
    for el in tree.iter():
        call = el.tag
        # system calls
        if call not in call_counter:
            call_counter[call] = 1
            call_counter[call + ' indicator'] = 1
        else:
            call_counter[call] += 1
        # bytes of data sent
        if call == 'send_socket':
            call_counter['bytes_sent'] += int(el.attrib['buffer_len'])
            call_counter['any_sent'] = 1
        # bytes of data received
        elif call == 'recv_socket':
            call_counter['bytes_received'] += int(el.attrib['buffer_len'])
            call_counter['any_received'] = 1
        # time spent in each process
        elif call == 'process':
            endtime = datetime.strptime(el.attrib['terminationtime'], '%M:%S.%f')
            starttime = datetime.strptime(el.attrib['starttime'], '%M:%S.%f')
            call_counter['totaltime'] += (endtime - starttime).total_seconds()
        # indicator variable for each socket number
        if 'socket' in el.attrib:
            if el.attrib['socket'] in features:
                call_counter[el.attrib['socket']] = 1
    return call_counter

## Feature extraction
def main():
    features = call_list + map(lambda s : s + ' indicator', call_list) + map(str, range(4000))\
               + ['bytes_sent', 'bytes_received', 'any_sent', 'any_received', 'totaltime']

    X, t, ids, features = create_data_matrix()
    print X.shape

    # indices of all nontrivial columns
    indices = [i for i in range(X.shape[1]) if X[:,i].any() and len(set(X[:,i])) > 1]
    X = X[:,indices]
    features = np.array(features)[indices]
    print 'Number of features:', len(features)

    pickle.dump((ids, X, t, features), open('data_matrix.p', 'w'))

if __name__ == "__main__":
    main()
