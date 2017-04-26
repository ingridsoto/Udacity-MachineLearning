#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time

from sklearn.metrics import accuracy_score

sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
#features_train = features_train[:len(features_train)//100]
#labels_train = labels_train[:len(labels_train)//100]


#########################################################
### your code goes here ###

from sklearn import svm
clf = svm.SVC(kernel='rbf', C=10000)
#clf=svm.SVC(kernel="linear")


t0=time()
clf.fit(features_train, labels_train)
print ("Training time:", round(time()-t0, 3), "s")

t1 = time()
pred = clf.predict(features_test)
print ("Prediction time:", round(time()-t1, 3), "s")
print ("Accuracy:", accuracy_score(pred, labels_test))


print (clf.predict(features_test[26]))

#########################################################


chris = []
# Get number of predicted emails written by Chris.  Ans: 877
for i in pred:
    if i == 1:
        chris.append(i)

print (len(chris))