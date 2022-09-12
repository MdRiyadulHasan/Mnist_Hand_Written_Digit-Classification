import dataset
import visualization
import prepare_data
import create_model
import train_model

if __name__ == '__main__':
    x_train,y_train,x_test,y_test = dataset.create_dataset()
    #visualization.draw_input_data(x_train,y_train)
    x_train, y_train, x_test, y_test = prepare_data.data_preparation(x_train,y_train,x_test,y_test)
    model=create_model.define_model()
    history=train_model.model_training(model,x_train,y_train,x_test,y_test)
    visualization.plot_loss_graph(history)
    visualization.plot_accuracy_graph(history)

    
