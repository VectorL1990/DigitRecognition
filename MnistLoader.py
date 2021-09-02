import matplotlib.pyplot as plt
import struct
import numpy as np
from sklearn.metrics import confusion_matrix
from datetime import timedelta
import os
import cv2

class MnistLoader(object):
    def __init__(self, in_image_width, in_image_height, in_num_channels):
        self.image_width = in_image_width
        self.image_height = in_image_height
        self.num_channels = in_num_channels


    def checkFolder(self, folder):
        if not os.path.exists(folder):
            os.mkdir(folder)
            print(folder)
        else:
            if not os.path.isdir(folder):
                os.mkdir(folder)

    def loadImage(self, export_dir, img_ubyte, label_ubyte):
        self.checkFolder(export_dir)
        images = self.decode_idx3_ubyte(img_ubyte)
        labels = self.decode_idx1_ubyte(label_ubyte)

        return images, labels


    def parserMnistData(self, data_dir):
        train_dir = os.path.join(data_dir, 'train')
        train_img_dir = os.path.join(data_dir, 'train-images.idx3-ubyte')
        train_label_dir = os.path.join(data_dir, 'train-labels.idx1-ubyte')
        self.train_images, self.train_labels = self.loadImage(train_dir, train_img_dir, train_label_dir)

        test_dir = os.path.join(data_dir, 'test')
        test_img_dir = os.path.join(data_dir, 't10k-images.idx3-ubyte')
        test_label_dir = os.path.join(data_dir, 't10k-labels.idx1-ubyte')
        self.test_images, self.test_labels = self.loadImage(test_dir, test_img_dir, test_label_dir)

    def decode_idx3_ubyte(self, idx3_ubyte_file):
        bin_file = open(idx3_ubyte_file, 'rb')
        file_data = bin_file.read()

        offset = 0
        fmt_header = '>IIII'
        # Read 4 integer data, initial read position should be offset(0)
        # 1st parameter of unpack_from represents reading stride
        # 3rd parameter of unpack_from represents reading initial position
        magic_num, num_images, self.image_height, self.image_width = struct.unpack_from(fmt_header, file_data, offset)
        print('idx3 magic num is: {}, image num is: {}'.format(magic_num, num_images))
        offset += struct.calcsize(fmt_header)
        # fmt_image represents image size
        # '>' represents right move
        # 'B' represents byte
        fmt_image = '>' + str(self.image_height*self.image_width) + 'B'

        images = struct.unpack_from('>'+str(num_images*self.image_height*self.image_width)+'B', file_data, struct.calcsize('>IIII'))
        bin_file.close()
        images = np.reshape(images, [num_images, self.image_height, self.image_width, self.num_channels])

        return images


    def decode_idx1_ubyte(self, idx1_ubtye_file):
        bin_file = open(idx1_ubtye_file, 'rb')
        file_data = bin_file.read()

        offset = 0
        fmt_header = '>II'
        magic_num, num_items = struct.unpack_from(fmt_header, file_data, offset)
        print('idx1 magic num is: {}, label num is: {}'.format(magic_num, num_items))

        labels = struct.unpack_from('>'+str(num_items)+'B', file_data, struct.calcsize('>II'))
        bin_file.close()
        labels = np.reshape(labels, [num_items])

        return labels


    def plotImages(self, images, classification_prediction, plot_num):
        fig=plt.figure(figsize=(8,8))
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
        for i in range(plot_num):
            plot_image = np.reshape(images[i], [self.image_height,self.image_width])
            ax=fig.add_subplot(6,5,i+1,xticks=[],yticks=[])
            ax.imshow(plot_image,cmap=plt.cm.binary,interpolation='nearest')
            ax.text(0,7,str(classification_prediction[i]))
        plt.show()

    def LoadAndPlot(self, plot_num):
        self.parserMnistData('data\MNIST')
        #plot_images = test_images[0:plot_num]
        #plot_labels = test_labels[0:plot_num]
        self.plotImages(self.test_images, self.test_labels, plot_num)

if __name__ == "__main__":
    app = MnistLoader(28,28,1)
    app.LoadAndPlot(9)

