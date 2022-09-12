from tensorflow import keras
from keras.utils import to_categorical
def data_preparation(x_train,y_train,x_test,y_test):
    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')
    x_train=x_train/255.0
    x_test=x_test/255.0
    y_train=to_categorical(y_train)
    y_test=to_categorical(y_test)
    return x_train, y_train, x_test, y_test
   
