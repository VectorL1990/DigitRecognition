import tensorflow as tf
import numpy as np
from TrainDigitTF2 import CNN
from TrainDigitTF2 import Train
from MnistLoader import MnistLoader
import matplotlib.pyplot as plt

class Predict(object):
    def __init__(self, in_img_size, in_num_channel, in_filter_size, in_max_pooling_size):
        self.mnist_loader = MnistLoader(in_img_size, in_img_size, in_num_channel)
        self.mnist_loader.parserMnistData('data\MNIST')

        latest_checkpoint = tf.train.latest_checkpoint('./ckpt')
        self.cnn = CNN()
        self.cnn.constructLayers(in_num_channel, in_filter_size, in_img_size, in_max_pooling_size)
        self.cnn.model.load_weights(latest_checkpoint)

    def predict(self, image_nb):
        img = self.mnist_loader.test_images[image_nb]
        
        x = np.array([img])
        print("shape of origin img is: {}".format(img.shape))
        print("shape of test img is: {}".format(x.shape))
        y = self.cnn.model.predict(x)
        print(y[0])
        print('  -> Predict picture number is: ', np.argmax(y[0]))
        print("shape of yis: {}".format(y.shape))

        fig=plt.figure(figsize=(8,8))
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
        plot_image = np.reshape(img, [self.mnist_loader.image_height,self.mnist_loader.image_width])
        ax=fig.add_subplot(6,5,1,xticks=[],yticks=[])
        ax.imshow(plot_image,cmap=plt.cm.binary,interpolation='nearest')
        ax.text(0,7,str(self.mnist_loader.test_labels[1]))
        plt.show()
        


if __name__ == "__main__":
    app = Predict(28, 1, 3, 2)
    app.predict(100)
