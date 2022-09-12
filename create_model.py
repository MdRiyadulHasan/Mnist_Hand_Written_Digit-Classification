from tensorflow import keras
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dense,BatchNormalization,Flatten
def define_model():
    model=Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten( ))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.summary()
    plot_model(model, to_file='models/my_model.png', show_shapes=True, dpi=600)
    return model
    
