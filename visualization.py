from matplotlib import pyplot as plt

def draw_input_data(x_train,y_train):
    plt.figure(figsize=(5,5))
    row=3
    col=3
    for i in range(row*col):
        plt.subplot(row,col,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_train[i], cmap=plt.get_cmap('Blues'))
        plt.xlabel(y_train[i])
        plt.savefig('figures/some_input_data.png', dpi=600)
    plt.show()
def plot_accuracy_graph(history):
    plt.figure(figsize=(5,5))
    plt.xticks([])
    plt.yticks([])
    plt.plot(history.history['accuracy'], color='blue', label='accuracy')
    plt.plot(history.history['val_accuracy'],color='orange', label='val_accuracy')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('figures/train_accuracy_graph.png', dpi=600)
    plt.show()
def plot_loss_graph(history):
    plt.figure(figsize=(5,5))
    plt.xticks([])
    plt.yticks([])
    plt.plot(history.history['loss'], color='blue', label='loss')
    plt.plot(history.history['val_loss'],color='orange', label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('figures/train_loss_graph.png', dpi=600)
    plt.show()
