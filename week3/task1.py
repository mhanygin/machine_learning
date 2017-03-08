from sklearn.svm import SVC
import numpy as np
import pandas

data = pandas.read_csv('./data/svm-data.csv')
Y = np.array(data.get('f1'))
print Y
X = np.array(zip(data.get('f2'), data.get('f3')))
print X
clf = SVC(C=100000, kernel='linear', random_state=241)
res = clf.fit(X, Y)
print res.support_
print res.support_vectors_
