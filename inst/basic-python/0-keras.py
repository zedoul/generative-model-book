#!/bin/python2

# http://blog.fastforwardlabs.com/2016/02/24/hello-world-in-keras-or-scikit-learn-versus.html

import seaborn as sns

# pandas.core.frame.DataFrame
iris = sns.load_dataset('iris')
print iris.head()

# numpy.ndarray
X = iris.values[:, 0:4] # select 1:3
y = iris.values[:, 4] # select 4


# scikit-learn::train_test_split()
from sklearn.cross_validation import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    train_size = 0.5,
                                                    random_state = 0)

# scikit-learn::linear_model
from sklearn.linear_model import LogisticRegressionCV
lr = LogisticRegressionCV()
lr.fit(train_X, train_y)

pred_y = lr.predict(test_X)
print "fraction correct (Accuracy) = {:.2f}".format(lr.score(test_X, test_y))

import numpy as np
def one_hot_encode_object_array(arr):
    from keras.utils import np_utils
    '''One hot encode a numpy array of objects (e.g. strings)'''
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

train_y_one = one_hot_encode_object_array(train_y)
test_y_one = one_hot_encode_object_array(test_y)


# In this case, we’ll build an extremely simple network:
# 4 features in the input layer (the four flower measurements),
# 3 classes in the ouput layer (corresponding to the 3 species),
# and 16 hidden units because (from the point of view of a GPU, 16 is a round number!)

from keras.models import Sequential
from keras.layers.core import Dense, Activation

model = Sequential()
model.add(Dense(16, input_shape=(4, )))
model.add(Activation('sigmoid'))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', metrics=['mae', 'acc'], optimizer = 'adam')
model.fit(train_X, train_y_one, verbose=0, batch_size=1)

loss = model.evaluate(test_X, test_y_one)

v = model.predict(test_X)

print "Test fraction correct (??) = {:.2f}".format(loss)

