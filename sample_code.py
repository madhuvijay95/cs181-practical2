import os
from datetime import datetime
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
import cPickle as pickle
import sys
from scikits.statsmodels.distributions import ECDF # installed from http://scikits.appspot.com/statsmodels

import util

# list of all XML tags
call_list = pickle.load(open('tag_list.p', 'r'))
call_set = set(call_list)

# List of all features (excluding the ECDF features, which are appended later). Note that the numbered features just
# refer to socket numbers.
features = call_list + map(lambda s : s + ' indicator', call_list) + map(str, range(4000))\
           + ['has_socket', 'socket count', 'has_import', 'has_export', 'FILE_ANY_ACCESS', 'SECURITY_ANONYMOUS',
              'destroy_window', 'targetpid count', 'delete_value count', 'bytes_sent', 'bytes_received', 'any_sent',
              'any_received', 'totaltime']

# Construct a data matrix for the first <num> observations of the training and test data.
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

        # use call_feats to generate features
        this_row_dict = call_feats(tree)
        # string containing the text of the file
        str = open(os.path.join(direc, datafile), 'r').read()
        # generate several features by searching the text of the file for certain substrings
        this_row_dict['has_socket'] = int('socket' in str)
        this_row_dict['has_import'] = int('import' in str)
        this_row_dict['has_export'] = int('export' in str)
        this_row_dict['socket count'] = str.count('socket')
        this_row_dict['FILE_ANY_ACCESS'] = int('FILE_ANY_ACCESS' in str)
        this_row_dict['SECURITY_ANONYMOUS'] = int('SECURITY_ANONYMOUS' in str)
        this_row_dict['destroy_window'] = int('destroy_window' in str)
        this_row_dict['targetpid count'] = str.count('targetpid')
        this_row_dict['delete_value count'] = str.count('delete_value')

        # convert the dictionary into an array
        this_row = np.array([(this_row_dict[feature] if feature in this_row_dict else 0) for feature in features])
        # add the row to the data matrix X
        if X is None:
            X = this_row
        else:
            X = np.vstack((X, this_row))

    # add empirical CDF features for every existing nontrivial non-binary feature
    X = X.T
    for i in range(len(X)):
        if X[i].any() and len(set(X[i])) > 2:
            X = np.vstack((X, ECDF(X[i])(X[i])))
            features.append(features[i] + ' ECDF')
    X = X.T

    return X, np.array(classes), ids, features

# Generate a dictionary containing most of the features for the input XML tree.
def call_feats(tree):
    call_counter = {}
    call_counter['bytes_sent'] = 0
    call_counter['bytes_received'] = 0
    call_counter['any_sent'] = 0
    call_counter['any_received'] = 0
    call_counter['totaltime'] = 0
    for el in tree.iter():
        call = el.tag
        # check for each system calls
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
