# -*- coding: utf-8 -*-
"""
@authors: Arnau Colom & David Moreno
"""    
import numpy as np
import sklearn
import graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import datasets

FILE_train1 = "vehicle.txt"
FILE_train2 = "segment.txt"
test_size = 0.3
depth_constraint = 5

X, y =  datasets.load_svmlight_file(FILE_train2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

clf = tree.DecisionTreeClassifier(max_depth=depth_constraint)
clf = clf.fit(X_train, y_train)
print('Accuracy: ',clf.score(X_test, y_test))
print('Error: ', 1-clf.score(X_test, y_test))
tree.plot_tree(clf) 

dot_data = tree.export_graphviz(clf, out_file=None, 
                                filled=True, rounded=True,  
                                special_characters=True)
graph = graphviz.Source(dot_data) 
graph.render("segment_tree") 
