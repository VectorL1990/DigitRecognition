import os

from MnistLoader import MnistLoader
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from PIL import Image


class CNN(object):
    def constructLayers(self, 
                        in_num_channel,
                        in_filter_size,
                        in_image_size,
                        in_max_pooling_size):

        self.num_channel = in_num_channel
        self.filter_size = in_filter_size
        self.image_size = in_image_size
        self.max_pooling_size = in_max_pooling_size

        model = models.Sequential()

        model.add(layers.Conv2D(32, 
                                (self.filter_size, self.filter_size), 
                                activation='relu', 
                                input_shape=(self.image_size, self.image_size, self.num_channel)))

        model.add(layers.MaxPooling2D((self.max_pooling_size, self.max_pooling_size)))

        model.add(layers.Conv2D(64,
                                (self.filter_size, self.filter_size),
                                activation='relu'))

        model.add(layers.MaxPooling2D((self.max_pooling_size, self.max_pooling_size)))

        model.add(layers.Conv2D(64,
                                (self.filter_size, self.filter_size),
                                activation='relu'))

        model.add(layers.Flatten())

        model.add(layers.Dense(64, activation='relu'))

        model.add(layers.Dense(10, activation='softmax'))

        model.summary()

        self.model = model


class Train(object):
    def __init__(self, 
                in_num_channel,
                in_filter_size,
                in_image_size,
                in_max_pooling_size):
        self.cnn = CNN()
        self.cnn.constructLayers(in_num_channel= in_num_channel,
                                in_filter_size= in_filter_size,
                                in_image_size= in_image_size,
                                in_max_pooling_size= in_max_pooling_size)
        self.mnist_loader = MnistLoader(in_image_size, in_image_size, in_num_channel)
        self.mnist_loader.parserMnistData('data\MNIST')

    # Use keras.callbacks.ModelCheckpoint to save model trained
    def train(self):
        check_path = './ckpt/cp-{epoch:04d}.ckpt'
        save_model_callback = tf.keras.callbacks.ModelCheckpoint(check_path,
                                                                save_weights_only = False,
                                                                verbose = 1,
                                                                period = 5)

        self.cnn.model.compile(optimizer='adam',
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])

        self.cnn.model.fit(self.mnist_loader.train_images, self.mnist_loader.train_labels,
                            epochs=5, callbacks=[save_model_callback])

        test_loss, test_acc = self.cnn.model.evaluate(self.mnist_loader.test_images, self.mnist_loader.test_labels)
        print("Accuracy is: {0}, and total amount of test images is: {1}".format(test_acc, len(self.mnist_loader.test_labels)))

if __name__ == "__main__":
    train_obj = Train(1, 3, 28, 2)
    train_obj.train()