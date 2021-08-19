import matplotlib.pyplot as plt
import struct
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

import sys
import os
import urllib.request
import tarfile
import zipfile
import cv2

image_width = 0
image_height = 0


def checkFolder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
        print(folder)
    else:
        if not os.path.isdir(folder):
            os.mkdir(folder)

def loadImage(export_dir, img_ubyte, label_ubyte):
    checkFolder(export_dir)
    images = decode_idx3_ubyte(img_ubyte)
    labels = decode_idx1_ubyte(label_ubyte)

    '''
    nums = len(labels)
    for i in range(nums):
        img_dir = os.path.join(export_dir, str(labels[i]))
        checkFolder(img_dir)
        img_file = os.path.join(img_dir, str(i)+'.png')
        imarr = images[i]
        cv2.imwrite(img_file, imarr)
    '''

    return images, labels


def parserMnistData(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    train_img_dir = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_label_dir = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    train_images, train_labels = loadImage(train_dir, train_img_dir, train_label_dir)

    test_dir = os.path.join(data_dir, 'test')
    test_img_dir = os.path.join(data_dir, 't10k-images.idx3-ubyte')
    test_label_dir = os.path.join(data_dir, 't10k-labels.idx1-ubyte')
    test_images, test_labels = loadImage(test_dir, test_img_dir, test_label_dir)

    return train_images, train_labels, test_images, test_labels

def decode_idx3_ubyte(idx3_ubyte_file):
    bin_file = open(idx3_ubyte_file, 'rb')
    file_data = bin_file.read()

    offset = 0
    fmt_header = '>IIII'
    # Read 4 integer data, initial read position should be offset(0)
    # 1st parameter of unpack_from represents reading stride
    # 3rd parameter of unpack_from represents reading initial position
    global image_height, image_width
    magic_num, num_images, image_height, image_width = struct.unpack_from(fmt_header, file_data, offset)
    print('idx3 magic num is: {}, image num is: {}'.format(magic_num, num_images))
    offset += struct.calcsize(fmt_header)
    # fmt_image represents image size
    # '>' represents right move
    # 'B' represents byte
    fmt_image = '>' + str(image_height*image_width) + 'B'

    '''
    # Create images array which is 3 dimensions
    images = np.empty((num_images, image_height, image_width))
    for i in range(num_images):
        img = struct.unpack_from(fmt_image, file_data, offset)
        images[i] = np.array(img).reshape((image_height, image_width))
        offset += struct.calcsize(fmt_image)
    '''

    images = struct.unpack_from('>'+str(num_images*image_height*image_width)+'B', file_data, struct.calcsize('>IIII'))
    bin_file.close()
    images = np.reshape(images, [num_images, image_height*image_width])

    return images


def decode_idx1_ubyte(idx1_ubtye_file):
    bin_file = open(idx1_ubtye_file, 'rb')
    file_data = bin_file.read()

    offset = 0
    fmt_header = '>II'
    magic_num, num_items = struct.unpack_from(fmt_header, file_data, offset)
    print('idx1 magic num is: {}, label num is: {}'.format(magic_num, num_items))
    '''
    labels = []
    
    fmt_label = '>B'
    for i in range(num_items):
        labels.append(struct.unpack_from(fmt_label, file_data, offset)[0])
        offset += struct.calcsize(fmt_label)
    '''
    labels = struct.unpack_from('>'+str(num_items)+'B', file_data, struct.calcsize('>II'))
    bin_file.close()
    labels = np.reshape(labels, [num_items])

    return labels


def plotImages(images, classification_prediction):
    '''
    
    assert len(images) == len(classification_prediction)

    fig, axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    global image_height, image_width

    # ax represent 
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(image_height, image_width), cmap='binary')

        xlabel = "True: {0}".format(classification_prediction[i])

        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])
    '''
    global image_height, image_width
    fig=plt.figure(figsize=(8,8))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
    for i in range(30):
        plot_image = np.reshape(images[i], [image_height,image_width])
        ax=fig.add_subplot(6,5,i+1,xticks=[],yticks=[])
        ax.imshow(plot_image,cmap=plt.cm.binary,interpolation='nearest')
        ax.text(0,7,str(classification_prediction[i]))
    plt.show()

def main(plot_num):
    train_images, train_labels, test_images, test_labels = parserMnistData('data\MNIST')
    #plot_images = test_images[0:plot_num]
    #plot_labels = test_labels[0:plot_num]
    plotImages(train_images, train_labels)


if __name__ == '__main__':
    main(9)
