#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from datetime import timedelta

test_images = [None, [None]]
train_images = [None, [None]]
test_class = [None]
test_batch_size = 256
train_batch_size = 64
fc_size = 128
num_classification = 0
img_size = 0
num_channels = 0
filter_size1 = 0
num_filters1 = 0
filter_size2 = 0
num_filters2 = 0
session = None
num_iterations = 0



def initGloabl(in_test_images, 
                in_train_images, 
                in_test_class, 
                in_test_batch_size,
                in_train_batch_size,
                in_fc_size,
                in_num_classification,
                in_img_size,
                in_num_channels,
                in_filter_size1,
                in_num_filters1,
                in_filter_size2,
                in_num_filters2,
                in_num_iterations):
    global test_images
    global train_images
    global test_class
    global test_batch_size
    global train_batch_size
    global fc_size
    global num_classification
    global img_size
    global num_channels
    global filter_size1
    global filter_size2
    global num_filters1
    global num_filters2
    global num_iterations
    test_images = in_test_images
    train_images = in_train_images
    test_class = in_test_class
    test_batch_size = in_test_batch_size
    train_batch_size = in_train_batch_size
    fc_size = in_fc_size
    num_classification = in_num_classification
    img_size = in_img_size
    num_channels = in_num_channels
    filter_size1 = in_filter_size1
    filter_size2 = in_filter_size2
    num_filters1 = in_num_filters1
    num_filters2 = in_num_filters2
    num_iterations = in_num_iterations


# This function generate weights for each filter by normal distribution
# Our target is to find out the best distribution of weights to classify all datas
def newWeights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.5))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

# For image analysis, input dimension should be 4
# Image number, same as input.
# Y-axis of each image. If 2x2 pooling is used, then the height and width of the input images is divided by 2.
# X-axis of each image. Ditto.
# Channels produced by the convolutional filters.
def convLayer(input,
                num_input_channels,
                filter_size,
                num_filters,
                use_pooling=True):
    print('convLayer: number of input is: {0}'.format(input.get_shape()))
    
    '''
    for example, there are 3 filters, each size is 2x2, but only outpu 1 channel
    |3  0|  |0  1|  |1   -1|
    |1  2|  |3  3|  |-1  -1|
    '''
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    weights = newWeights(shape=shape)

    biases = new_biases(length=num_filters)

    '''
    For 2D image, layer should be 4 dimensions
    1st dimension represents number of images,
    2nd dimension represents image height
    3rd dimension represents image width
    4th dimension represents number of channel, which means a feature
    '''
    layer = tf.nn.conv2d(input = input,
                        filter = weights,
                        strides = [1, 1, 1, 1],
                        padding = 'SAME')
    
    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value = layer,
                                ksize = [1, 2, 2, 1],
                                strides = [1, 2, 2, 1],
                                padding = 'SAME')

    layer = tf.nn.relu(layer)

    return layer, weights


def flattenLayer(layer):
    
    # Let's assume the input layer is conv layer
    # Then layer shape should represet
    # 1D = number of images, 2D = image height, 3D = image width, 4D = number of channels
    layer_shape = layer.get_shape()

    # Let's assume each original image has 16 channels
    # Which means there are 16 "generated images" of each original image
    # But now we want to "flatten" all channels
    # which means that we want to assemble all "generated images" into a single array
    # In that case we generate an 1D array which contains 16 images' pixels
    print('flattenLayer: shape of input layer is: {0}'.format(layer_shape))
    image_height = layer_shape[1]
    image_width = layer_shape[2]
    num_channels = layer_shape[3]
    num_features = image_width*image_height*num_channels

    # Here we flatten image array
    # 1st D represents number of images, so we transfer -1 as 1st parameter
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


def fullyConnectLayer(input,
                        num_inputs,
                        num_outputs,
                        use_relu=True):
    weights = newWeights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def constructLayersAndOptimizer():
    global img_size
    global num_channels
    global filter_size1
    global filter_size2
    global num_filters1
    global num_filters2
    global num_classification
    global session

    img_size_flat = img_size*img_size
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name = 'x')

    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

    y_true = tf.placeholder(tf.float32, shape=[None, num_classification], name = 'y_true')

    y_true_classification = tf.argmax(y_true, dimension = 1)

    layer_conv1, weights_conv1 = convLayer(input = x_image, 
                                            num_input_channels= num_channels,
                                            filter_size= filter_size1,
                                            num_filters= num_filters1,
                                            use_pooling=True)

    layer_conv2, weights_conv2 = convLayer(input=layer_conv1,
                                            num_input_channels=num_filters1,
                                            filter_size=filter_size2,
                                            num_filters=num_filters2,
                                            use_pooling=True)

    # For 2D image trainings, layer_flat is a 2D array
    # 1st D represents number of images
    # 2nd D represents assembled pixels of all generated channel images
    # num_features might be 28*28*16(we assume there are 16 channels being extracted)
    layer_flat, num_features = flattenLayer(layer_conv2)

    # For 2D image training, just for example, dimension of layer_flat is 60000x(28*28*16)
    # Then dimension of layer_fc1 is 60000x128
    layer_fc1 = fullyConnectLayer(input= layer_flat,
                                    num_inputs=num_features,
                                    num_outputs=fc_size,
                                    use_relu=True)


    layer_fc2 = fullyConnectLayer(input=layer_fc1,
                                    num_inputs=fc_size,
                                    num_outputs=num_classification,
                                    use_relu=False)

    # softmax calculate probabilities on each dimensions
    # by default it do softmax calculation on the last dimension
    y_predict = tf.nn.softmax(layer_fc2)

    y_predict_classification = tf.argmax(y_predict, dimension = 1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels = y_true)

    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    corret_prediction = tf.equal(y_predict_classification, y_true_classification)

    accuracy = tf.reduce_mean(tf.cast(corret_prediction, tf.float32))

    session = tf.Session()

    session.run(tf.global_variables_initializer())

    return optimizer, accuracy, y_predict_classification, y_true_classification, x, y_true


def optimize(num_iterations, flat_size_imgs, true_classifications, optimizer, accuracy, x, y_true):
    global train_batch_size
    global session

    start_time = time.time()

    for i in range(num_iterations):
        # We should assign images to x and classification here
        x_batch = flat_size_imgs[0 : train_batch_size]
        y_true_batch = true_classifications[0 : train_batch_size]
        x_batch_array = x_batch.eval(session= session)
        y_true_batch_array = y_true_batch.eval(session = session)
        feed_dict_train = {x: x_batch_array, y_true: y_true_batch_array}
        session.run(optimizer, feed_dict = feed_dict_train)

        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict = feed_dict_train)

            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            print(msg.format(i + 1, acc))

    end_time = time.time()

    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def plotExampleErrors(predict_classification, correct, test_imgs, test_classification):
    incorrect = (correct == False)
    images = test_imgs[incorrect]
    predict_classification = predict_classification[incorrect]
    true_classification = test_classification[incorrect]

    plotImage(images=images, true_class=true_classification, predict_class= predict_classification)


def printTestAccuracy(show_example_errors, y_predict_classification, x, y_true):
    global test_images
    global test_class
    global test_batch_size
    
    num_test_images = len(test_images)
    predict_class = np.zeros(shape=num_test_images, dtype=np.int)
    i = 0

    while i < num_test_images:
        j = min(i + test_batch_size, num_test_images)
        images = test_images[i:j, : ]
        labels = test_class[i:j, : ]
        feed_dict = {x: images, y_true: labels}
        predict_class[i:j] = session.run(y_predict_classification, feed_dict=feed_dict)
        i = j

    correct_class_list = (test_class == predict_class)

    correct_sum = correct_class_list.sum()

    acc = float(correct_sum) / num_test_images

    print("Accuracy is: {0:.1%} = ({1} / {2})".format(acc, correct_sum, num_test_images))

    if show_example_errors:
        plotExampleErrors(predict_classification=predict_class, correct=correct_class_list)


def plotConvWeights(weights, input_channel=0):
    w = session.run(weights)
    w_min = np.min(w)
    w_max = np.max(w)

    num_filters = w.shape[3]

    num_grids = math.ceil(math.sqrt(num_filters))

    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = w[:, :, input_channel, i]
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def plot_conv_layer(layer, image):
    feed_dict = {x: [image]}
    values = session.run(layer, feed_dict=feed_dict)
    num_filters = values.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = values[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap='binary')

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def plotImage(image):
    global img_size
    img_shape = (img_size, img_size)
    plt.imshow(image.reshape(img_shape),
                interpolation='nearest',
                cmap='binary')
    plt.show()


def RunTrain():
    global num_iterations
    global train_images
    img_flat = tf.reshape(train_images, [-1, img_size*img_size])
    optimizer, accuracy, y_predict_classification, y_true_classification, x, y_true = constructLayersAndOptimizer()
    optimize(num_iterations=num_iterations, 
            flat_size_imgs=img_flat, 
            true_classifications= y_true_classification,
            optimizer= optimizer,
            accuracy= accuracy,
            x= x,
            y_true= y_true)

    printTestAccuracy(True, y_predict_classification=y_predict_classification, x= x, y_true= y_true)