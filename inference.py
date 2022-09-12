from tensorflow import keras
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
def prepare_image(img_name):
    img=img_to_array(img_name)
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    img = img.astype('float32')
    img = img / 255.0
    return img
if __name__=="__main__":
    model_name='models\my_model.h5'
    model=load_model(model_name)
    img_name=cv2.imread('figures\sample_image.png',0)
    img_name=cv2.resize(img_name,(28,28))
    print(img_name.shape)
    cv2.imshow('Inference_image',img_name)
    cv2.waitKey(5000)
    img=prepare_image(img_name)
    predict_value = model.predict(img)
    digit =np.argmax(predict_value)
    print(f'The given image is : {digit}')

