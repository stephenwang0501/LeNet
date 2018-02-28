import tensorflow as tf


def build_lenet(x, keep_prob):
    with tf.variable_scope(name_or_scope='LeNet', reuse=tf.AUTO_REUSE):
        print('LeNet structure:')

        net = tf.layers.conv2d(
            inputs=x,
            filters=32,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        print('Convolution 1: {}'.format(net.shape[1:]))

        net = tf.layers.batch_normalization(inputs=net)
        print('Batch normalization 1: {}'.format(net.shape[1:]))

        net = tf.layers.max_pooling2d(
            inputs=net,
            pool_size=2,
            strides=2)
        print('Max pool 1: {}'.format(net.shape[1:]))

        net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        print('Convolution 2: {}'.format(net.shape[1:]))

        net = tf.layers.batch_normalization(inputs=net)
        print('Batch normalization 2: {}'.format(net.shape[1:]))

        net = tf.layers.max_pooling2d(
            inputs=net,
            pool_size=2,
            strides=2)
        print('Max pool 2: {}'.format(net.shape[1:]))

        net = tf.layers.flatten(inputs=net)
        print('FC input: {}'.format(net.shape[1:]))

        net = tf.contrib.layers.fully_connected(
            inputs=net,
            num_outputs=1024)
        print('FC 1: {}'.format(net.shape[1:]))

        net = tf.layers.dropout(
            inputs=net,
            rate=keep_prob)
        print('Drop out: {}'.format(net.shape[1:]))

        net = tf.contrib.layers.fully_connected(
            inputs=net,
            num_outputs=84)
        print('FC 2: {}'.format(net.shape[1:]))

        net = tf.contrib.layers.fully_connected(
            inputs=net,
            num_outputs=10,
            activation_fn=None)
        print("FC 3: {}".format(net.shape[1:]))

    return net
