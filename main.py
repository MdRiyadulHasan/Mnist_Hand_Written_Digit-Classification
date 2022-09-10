
from unicodedata import name
import dataset
import visualization

if __name__ == '__main__':
    x_train,y_train,x_test,y_test = dataset.create_dataset()
    visualization.draw_input_data(x_train,y_train)

    
