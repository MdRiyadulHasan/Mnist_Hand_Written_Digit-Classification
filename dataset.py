import tensorflow
from tensorflow import keras
from keras.datasets import mnist
from matplotlib import pyplot as plt
def create_dataset():
    (x_train,y_train), (x_test,y_test)=mnist.load_data()
    x_train=x_train.reshape(x_train.shape[0],28,28,1)
    x_test=x_test.reshape(x_test.shape[0],28,28,1)
    return x_train, y_train,x_test,y_test

