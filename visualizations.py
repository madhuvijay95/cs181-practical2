import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

results = pickle.load(open('classifier_results.p', 'r'))
accuracies = zip(results.keys(), map(lambda tup : tup[2], results.values()))
accuracies.sort(key = lambda tup : -tup[1])
accuracies = [(k,v) if 'LogisticRegression' not in k else (k.replace('LogisticRegression','LogReg'),v)
              for k,v in accuracies]

width = 0.5
plt.bar(np.arange(len(results)), [tup[1] for tup in accuracies], width, color='r')
plt.ylabel('Accuracy rate')
plt.title('Accuracy by model')
plt.xticks(np.arange(len(results)) + width/2, [tup[0].split('(')[0][:-1] if '(' in tup[0] else tup[0] for tup in accuracies])
plt.get_current_fig_manager().window.showMaximized()
plt.savefig('model_accuracies.png')