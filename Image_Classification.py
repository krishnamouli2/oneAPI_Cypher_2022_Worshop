from pytictoc import TicToc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

(train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

train_images.shape,train_labels.shape
train_images=train_images.reshape(60000,784)
X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size = 0.3, random_state = 1234)
label_dictionnary = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 
                     3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 
                     7:'Sneaker', 8:'Bag', 9:'Ankle boot' }
def true_label(x):
    return label_dictionnary[x]

X_train = X_train / 255.
X_test = X_test / 255.

t = TicToc()
t.tic()
clf = MLPClassifier(hidden_layer_sizes=(512,512), max_iter=#No. of Epochs#)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
perf = accuracy_score(y_test, y_pred)
print(perf)
t.toc()
