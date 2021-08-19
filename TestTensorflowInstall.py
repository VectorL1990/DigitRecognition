import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

v_1 = tf.constant([1,2,3,4])
v_2 = tf.constant([2,1,5,1])
v_add = tf.add(v_1, v_2)

with tf.Session() as sess:
    print(sess.run(v_add))

sess.close()