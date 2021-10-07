import numpy as np
import scipy 
import sklearn
import graphviz
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import cross_val_score
from matplotlib import cm

from sklearn import svm


FILE_train = "phishing.txt"

X, y =  sklearn.datasets.load_svmlight_file(FILE_train)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



gamma_vec = np.logspace(-4, 3, 7,endpoint=True)
#C_vec = np.logspace(-4, 3, 7,endpoint=True)
C_vec = 1
score_mean = np.zeros((7,7))
score_std = np.zeros((7,7))


for i in range(0,len(gamma_vec)):
    for j in range(0,len(C_vec)):
        
        clf = svm.SVC(kernel = 'rbf', gamma = gamma_vec[i], C = C_vec[j],probability = False)
        #clf.fit(X_train, y_train)
        #score[int(i),int(j)] = clf.score(X_test,y_test)
        scores = cross_val_score(clf, X, y, cv=5)
        score_mean[int(i),int(j)] = scores.mean()
        score_std[int(i),int(j)]  = scores.std()  
        print(i)
    
    
print(score_mean)


np.savetxt("array_mean.txt", score_mean, fmt="%s")

np.savetxt("array_std.txt", score_std, fmt="%s")
# Set up grid and test data
nx, ny = 7, 7
x = range(-4,3)
y = range(-4,3)

data = score_mean

hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')
ha.set_ylabel('C x10')
ha.set_xlabel('Gamma x10')
ha.set_zlabel('Score Mean')


X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
ha.plot_surface(X, Y, data, cmap=cm.plasma)

plt.show()


# Set up grid and test data
nx, ny = 7, 7
x = range(-4,3)
y = range(-4,3)

data = score_std

hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')
ha.set_ylabel('C x10')
ha.set_xlabel('Gamma x10')
ha.set_zlabel('Score Variance')


X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
ha.plot_surface(X, Y, data, cmap=cm.plasma)

plt.show()