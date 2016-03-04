import cPickle as pickle
import numpy as np

import util

from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import itertools

from sknn.mlp import Classifier, Layer
#import skflow

import matplotlib.pyplot as plt


ids, X, t, features = pickle.load(open('data_matrix.p', 'r'))
print X.shape
X = X[t != -1]
t = t[t != -1]
print X.shape

train_list = np.random.choice(range(len(X)), size = 0.7 * len(X), replace=False)
train_mask = np.array([i in train_list for i in range(len(X))])
valid_mask = ~train_mask

def feature_select(clf):
    cvscore = np.mean(cross_val_score(clf, X, t))
    clf.fit(X, t)
    try:
        feature_importances = list(reversed(np.array(features)[np.argsort(clf.feature_importances_)]))
    except:
        feature_importances = None
    selection_results = {'mean' : dict(), 'median' : dict()}
    scalings = [0, 0.25, 0.5, 0.75, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2]
    for scaling in scalings:
        X_new = SelectFromModel(clf, threshold=str(scaling)+'*mean', prefit=True).transform(X)
        selection_results['mean'][scaling] = np.mean(cross_val_score(clf, X_new, t))
        X_new = SelectFromModel(clf, threshold=str(scaling)+'*median', prefit=True).transform(X)
        selection_results['median'][scaling] = np.mean(cross_val_score(clf, X_new, t))
    best_select = max(itertools.product(['mean', 'median'], scalings), key = lambda (m,s) : selection_results[m][s])
    model = SelectFromModel(clf, threshold=str(best_select[1]) + '*' + best_select[0], prefit=True)
    X_new = model.transform(X)
    feature_mask = model.get_support()
    cvscore_selected = np.mean(cross_val_score(clf, X_new, t))
    clf.fit(X_new, t)
    return cvscore, feature_importances, best_select, feature_mask, cvscore_selected, clf

def valid_accuracy(clf):
    clf.fit(X[train_mask], t[train_mask])
    predictions = clf.predict(X[valid_mask])
    return float(sum(np.array(predictions) == t[valid_mask])) / len(t[valid_mask])

results = dict()


clf = svm.SVC()
accuracy = valid_accuracy(clf)#np.mean(cross_val_score(clf, X, t))
clf.fit(X, t)
results['SVC'] = (clf, range(X.shape[1]), accuracy)
print 'SVC:', accuracy

clf = svm.LinearSVC()
accuracy = valid_accuracy(clf)#np.mean(cross_val_score(clf, X, t))
clf.fit(X, t)
results['LinearSVC'] = (clf, range(X.shape[1]), accuracy)
print 'LinearSVC:', accuracy

sample_counts = [1, 2]
est_counts = [5, 10, 15, 20, 50, 100, 250, 500, 1000]
RF_results = dict()
for s, e in itertools.product(sample_counts, est_counts):
    print (s, e)
    # use 10-fold cross-validation because 5-fold cross-validation had a lot of variance in optimal parameters
    RF_results[(s,e)] = np.mean(cross_val_score(RandomForestClassifier(n_estimators=e, min_samples_split=10), X[train_mask], t[train_mask], cv=10))
#print list(reversed(np.argsort(RF_results.values())))
#print [RF_results.keys()[tup] for tup in reversed(np.argsort(RF_results.values()))]
#print [RF_results[RF_results.keys()[tup]] for tup in reversed(np.argsort(RF_results.values()))]
s_opt, e_opt = max(RF_results, key = lambda tup : RF_results[tup])

clf = RandomForestClassifier(n_estimators=e_opt, min_samples_split=s_opt)
_, _, _, feature_mask, _, clf = feature_select(clf)
accuracy = valid_accuracy(clf)
results['RandomForest (n_est=%d, min_samples_split=%d)' % (e_opt, s_opt)] = (clf, feature_mask, accuracy)
print 'Random Forest (n_est=%d, min_samples_split=%d):' % (e_opt, s_opt), accuracy
#print list(enumerate(reversed(np.array(features)[np.argsort(clf.feature_importances_)])))



sample_counts = [1, 2]
est_counts = [5, 10, 15, 20, 50, 100, 250, 500, 1000]
ET_results = dict()
for s, e in itertools.product(sample_counts, est_counts):
    print (s, e)
    # use 10-fold cross-validation because 5-fold cross-validation had a lot of variance in optimal parameters
    ET_results[(s,e)] = np.mean(cross_val_score(ExtraTreesClassifier(n_estimators=e, min_samples_split=10), X[train_mask], t[train_mask], cv=10))
#print list(reversed(np.argsort(ET_results.values())))
#print [ET_results.keys()[tup] for tup in reversed(np.argsort(ET_results.values()))]
#print [ET_results[ET_results.keys()[tup]] for tup in reversed(np.argsort(ET_results.values()))]
s_opt, e_opt = max(ET_results, key = lambda tup : ET_results[tup])
print s_opt, e_opt

clf = ExtraTreesClassifier(n_estimators=e_opt, min_samples_split=s_opt)
#clf = RandomForestClassifier(min_samples_split=1)
_, _, _, feature_mask, _, clf = feature_select(clf)
accuracy = valid_accuracy(clf)
results['ExtraTrees (n_est=%d, min_samples_split=%d)' % (e_opt, s_opt)] = (clf, feature_mask, accuracy)
print 'ExtraTrees (n_est=%d, min_samples_split=%d):' % (e_opt, s_opt), accuracy




DT_results = dict()
for s in [1,2,3]:
    # use 10-fold cross-validation because 5-fold cross-validation had a lot of variance in optimal parameters
    DT_results[s] = np.mean(cross_val_score(DecisionTreeClassifier(min_samples_split=s), X[train_mask], t[train_mask], cv=10))
s_opt = max(DT_results, key = lambda s : DT_results[s])
clf = DecisionTreeClassifier(min_samples_split=s_opt)
_, _, _, feature_mask, _, clf = feature_select(clf)
accuracy = valid_accuracy(clf)
results['Decision Tree (n_est=%d)' % s_opt] = (clf, feature_mask, accuracy)
print 'Decision Tree (n_est=%d):' % s_opt, accuracy


clf = LogisticRegression()
accuracy = valid_accuracy(clf)#np.mean(cross_val_score(clf, X, t))
clf.fit(X, t)
results['LogisticRegression'] = (clf, range(X.shape[1]), accuracy)
print 'Logistic Regression:', accuracy

clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
accuracy = valid_accuracy(clf)#np.mean(cross_val_score(clf, X, t))
clf.fit(X, t)
results['LogisticRegressionMultinomial'] = (clf, range(X.shape[1]), accuracy)
print 'Logistic Regression (Multinomial):', accuracy

clf = GaussianNB()
accuracy = valid_accuracy(clf)#np.mean(cross_val_score(clf, X, t))
clf.fit(X, t)
results['GaussianNB'] = (clf, range(X.shape[1]), accuracy)
print 'Gaussian NB:', accuracy

kNN_results = dict()
for k in [1, 5, 10, 20, 30, 40, 50]:
    kNN_results[k] = np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=k, weights='distance'), X[train_mask], t[train_mask]))
k_opt = max(kNN_results, key = lambda k : kNN_results[k])
clf = KNeighborsClassifier(n_neighbors=k_opt, weights='distance')
accuracy = valid_accuracy(clf)#np.mean(cross_val_score(clf, X, t))
clf.fit(X, t)
results['kNN (k=%d)' % k_opt] = (clf, range(X.shape[1]), accuracy)
print 'k-Nearest Neighbors (k=%d):' % k_opt, accuracy

best_model_name = max(results, key = lambda k : results[k][2])
best = results[best_model_name]
clf = best[0]
feature_mask = best[1]
print 'Best model: %s (accuracy %f)' % (best_model_name, results[best_model_name][2])
pickle.dump((feature_mask, clf), open('classifier.p', 'w'))
pickle.dump(results, open('classifier_results.p', 'w'))


#clf = skflow.TensorFlowLinearClassifier()
#clf.fit(X[train_mask], t[train_mask])
#predictions = clf.predict(X[valid_mask])
#print float(sum(np.array(predictions) == t[valid_mask])) / len(t[valid_mask])



#from sklearn.neural_network import MLPClassifier



#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation
#from keras.optimizers import SGD
#model = Sequential()
#model.add(Dense(64, input_dim=X.shape[1], init='uniform'))
#model.add(Activation('tanh'))
#model.add(Dropout(0.5))
#model.add(Dense(64, init='uniform'))
#model.add(Activation('tanh'))
#model.add(Dropout(0.5))
#model.add(Dense(10, init='uniform'))
#model.add(Activation('softmax'))
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)
#model.fit(X[train_mask], t[train_mask], nb_epoch=20, batch_size=16, show_accuracy=True)
#score = model.evaluate(X[valid_mask], t[valid_mask], batch_size=16)


#mlp = Classifier(layers = [Layer('Sigmoid', units=128), Layer('Tanh', units=64), Layer(type='Softmax')])
#mlp.fit(X[train_mask], t[train_mask])
#predictions = mlp.predict(X[valid_mask])[:,0]
#print float(np.sum(np.array(predictions) == t[valid_mask])) / len(t[valid_mask])