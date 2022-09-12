import dataset
import prepare_data
import confusion_matrix_classification_report
from tensorflow import keras
from keras.models import load_model


if __name__=='__main__':
    model_name='models\my_model.h5'
    x_train,y_train,x_test,y_test=dataset.create_dataset()
    x_train,y_train,x_test,y_test=prepare_data.data_preparation(x_train,y_train,x_test,y_test)
    model=load_model(model_name)
    _,acc=model.evaluate(x_test,y_test)
    print(f'Accuracy is : {round(acc*100,2)}')
    confusion_matrix_classification_report.draw_confusion_matrix_classification_report(model,x_test,y_test)