import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def draw_confusion_matrix_classification_report(model,x_test,y_test):
    y_pred=model.predict(x_test)
    y_pred=[np.argmax(i) for i in y_pred]
    y_pred=np.array(y_pred)
    y_test=[np.argmax(i) for i in y_test]
    cm=confusion_matrix(y_test,y_pred)
    print(cm)
    plt.figure(figsize=(7,7))
    sns.heatmap(cm, cbar=False, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('True_value')
    plt.ylabel('Predicted_value')
    plt.savefig('figures/confusion_matrix.png', dpi=600)  
    plt.show()

    print("\n Classification Report \n ")
    print(classification_report(y_test,y_pred))
