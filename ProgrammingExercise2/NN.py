# -*- coding: utf-8 -*-
"""
@authors: Arnau Colom & David Moreno
"""    
import numpy as np
# import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import datasets

FILE_train = 'phishing.txt'
alphas = np.logspace(-5, 2, 20)
X, y =  datasets.load_svmlight_file(FILE_train)

for i in range(1, 7):
    t_size = i*0.1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size)
    train_errors = []
    test_errors = []
    train_accuracy = []
    test_accuracy = []
    for a in alphas:
      nn_clf = MLPClassifier( alpha=a, activation = 'logistic',
                              hidden_layer_sizes=(16,8), 
                              max_iter=500, random_state=1)
    
      nn_clf.fit(X_train,y_train)
    
      test_prediction = nn_clf.predict(X_test)
      train_prediction = nn_clf.predict(X_train)
    
      train_accuracy.append(nn_clf.score(X_train, y_train))
      test_accuracy.append(nn_clf.score(X_test, y_test))
    
      train_errors.append(1-nn_clf.score(X_train, y_train))
      test_errors.append(1-nn_clf.score(X_test, y_test))

    print(t_size)
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].loglog(alphas, test_errors, '-r')
    axs[0, 0].set_title('MSE Test, test size = '+ str(round(t_size, 2)))
    
    axs[1, 0].loglog(alphas, test_accuracy, '-b')
    axs[1, 0].set_title('Accuracy Test, test size = '+ str(round(t_size, 2)))
    
    axs[0, 1].loglog(alphas, train_errors, '-r')
    axs[0, 1].set_title('MSE Train, test size = '+ str(round(t_size, 2)))
    
    axs[1, 1].loglog(alphas, train_accuracy, '-b')
    axs[1, 1].set_title('Accuracy Train, test size = '+ str(round(t_size, 2)))
    
    for ax in axs.flat:
        ax.set(xlabel= r'$\alpha$', ylabel='value')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    
