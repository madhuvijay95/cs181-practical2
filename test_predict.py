import cPickle as pickle
#from sample_code import create_data_matrix
import csv
import numpy as np

# load data
ids, X, t, used_features = pickle.load(open('data_matrix.p', 'r'))
# load optimal classifier and feature set
feature_mask, clf = pickle.load(open('classifier.p', 'r'))

# extract test data
X_test = X[t == -1][:, feature_mask]
print X_test.shape
ids_test = np.array(ids)[t == -1]
# refit optimal model on train data
clf.fit(X[t != -1][:, feature_mask], t[t != -1])
# predict on test data
predictions = clf.predict(X_test)
predictions = zip(ids_test, predictions)

# write to csv
with open('predictions.csv', 'wb') as output_file:
    writer = csv.writer(output_file)
    writer.writerow(('Id', 'Prediction'))
    for tup in predictions:
        writer.writerow(tup)