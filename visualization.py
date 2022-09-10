from matplotlib import pyplot as plt

def draw_input_data(x_train,y_train):
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
