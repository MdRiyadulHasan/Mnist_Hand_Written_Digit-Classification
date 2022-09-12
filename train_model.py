from gc import callbacks
from tabnanny import verbose
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn import metrics
def model_training(model, x_train,y_train,x_test,y_test):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    earlystop = EarlyStopping(monitor = 'val_accuracy',
                              min_delta = 0,
                              patience = 5,
                              verbose = 1,
                              mode='auto')
    checkpoint = ModelCheckpoint(filepath='models/my_model.h5',
                                monitor='val_accuracy',
                                mode='auto',
                                verbose=1,
                                save_best_only=True)
    callback_list=[earlystop,checkpoint]
    history= model.fit(x_train,y_train, epochs=5, validation_data=(x_test, y_test), verbose=1, callbacks=callback_list)
    return history