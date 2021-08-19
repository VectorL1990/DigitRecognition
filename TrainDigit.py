import tensorflow as tf
import numpy as np

test_images = [None, [None]]
train_images = [None, [None]]
test_class = [None]
predict_class = [None]
test_batch_size = 256

def initGloabl(in_test_images, in_train_images, in_test_class, in_predict_class):
    global test_images, train_images, test_class, predict_class
    test_images = in_test_images
    train_images = in_train_images
    test_class = in_test_class
    predict_class = in_predict_class


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
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    weights = newWeights(shape=shape)

    biases = new_biases(length=num_filters)

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
    layer_shape = layer.get_shape()

    img_height = layer_shape[1].num_elements()
    img_width = layer_shape[2].num_elements()
    num_channels = layer_shape[3].num_elements()

    num_features = img_width*img_height*num_channels

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

layer_flat, num_features = flattenLayer(layer_conv2)

layer_fc1 = fullyConnectLayer(input= layer_flat,
                                num_inputs=num_features,
                                num_outputs=fc_size,
                                use_relu=True)

layer_fc2 = fullyConnectLayer(input=layer_fc1,
                                num_inputs=fc_size,
                                num_outputs=num_classification,
                                use_relu=False)

y_predict = tf.nn.softmax(layer_fc2)

y_predict_classification = tf.argmax(y_predict, dimension = 1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels = y_true)

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

corret_prediction = tf.equal(y_predict_classification, y_true_classification)

accuracy = tf.reduce_mean(tf.cast(corret_prediction, tf.float32))

session = tf.Session()

session.run(tf.global_variables_initializer())

train_batch_size = 64

total_iterations = 0

def optimize(num_iterations, flat_size_imgs, true_classifications):
    global total_iterations
    global train_batch_size

    start_time = time.time()

    current_batch_start_location = 0
    for i in range(total_iterations, total_iterations + num_iterations):
        # We should assign images to x and classification here
        x_batch = flat_size_imgs[current_batch_start_location : current_batch_start_location + train_batch_size]
        y_true_batch = true_classifications[current_batch_start_location : current_batch_start_location + train_batch_size]
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        session.run(optimizer, feed_dict = feed_dict_train)

        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict = feed_dict_train)

            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            print(msg.format(i + 1, acc))

    total_iterations += num_iterations

    end_time = time.time()

    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def plotExampleErrors(predict_classification, correct, test_imgs, test_classification):
    incorrect = (correct == False)
    images = test_imgs[incorrect]
    predict_classification = predict_classification[incorrect]
    true_classification = test_classification[incorrect]

    plot_images(images=images, true_class=true_classification, predict_class= predict_classification)


def printTestAccuracy(show_example_errors=False, show_confusion_matrix=False):
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

