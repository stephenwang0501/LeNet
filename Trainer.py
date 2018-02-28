import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from random import shuffle
import LeNet


class LeNetTrainer:
    lenet = None

    x_train = None
    y_train = None

    x_validation = None
    y_validation = None

    x_test = None
    y_test = None

    batch_size = None
    epoch = None
    learn_rate = None
    train_keep_prob = None

    image_batch = []

    x_tensor = None
    y_tensor = None
    y_one_hot_tensor = None
    keep_prob_tensor = None

    initialized = False
    variable_scope_base = 'trainer'

    def __init__(self, learn_rate, batch_size, epoch, train_keep_prob):
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.train_keep_prob = train_keep_prob
        assert self.learn_rate > 0.0
        assert self.batch_size > 0
        assert self.epoch > 0
        assert 0.0 <= self.train_keep_prob <= 1.0

    def set_data(self):
        self.initialized = False

        mnist = input_data.read_data_sets("MNIST_data", reshape=False)
        self.x_train, self.y_train = mnist.train.images, mnist.train.labels
        self.x_validation, self.y_validation = mnist.validation.images, mnist.validation.labels
        self.x_test, self.y_test = mnist.test.images, mnist.test.labels

        assert len(self.x_train) == len(self.y_train)
        assert len(self.x_validation) == len(self.y_validation)
        assert len(self.x_test) == len(self.y_test)

        self.image_batch.append(None)
        self.image_batch += self.x_train[0].shape

        with tf.variable_scope(self.variable_scope_base):
            self.x_tensor = tf.placeholder(tf.float32, self.image_batch)
            self.y_tensor = tf.placeholder(tf.int32)
            self.keep_prob_tensor = tf.placeholder(tf.float32)
            self.y_one_hot_tensor = tf.one_hot(indices=tf.cast(self.y_tensor, tf.int32), depth=10)

        self.lenet = LeNet.build_lenet(x=self.x_tensor, keep_prob=self.keep_prob_tensor)

        self.initialized = True

    # use this function for testing
    def evaluation(self, x_data, y_data, accuracy_ops):
        n_validates = len(x_data)
        correct = 0
        sess = tf.get_default_session()
        for offset in range(0, n_validates, self.batch_size):
            x_batch, y_batch = x_data[offset:offset + self.batch_size], y_data[offset:offset + self.batch_size]
            accuracy = sess.run(accuracy_ops,
                                feed_dict={self.x_tensor: x_batch, self.y_tensor: y_batch, self.keep_prob_tensor: 1.0})
            correct += accuracy * len(x_batch)
        return correct / n_validates

    # use this function for training
    def train(self):
        if not self.initialized:
            print('Data is not initialized, call set_data first!')
            return

        with tf.variable_scope(self.variable_scope_base):
            print('Start training...')

            validation_x = []
            validation_y = []
            train_y = []

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.lenet, labels=self.y_one_hot_tensor)
            loss = tf.reduce_mean(cross_entropy)

            ops = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(loss)

            correct_predict = tf.equal(tf.argmax(self.lenet, 1), tf.argmax(self.y_one_hot_tensor, 1))
            accuracy_ops = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

            with tf.Session() as sess:
                n_samples = len(self.x_train)
                items = [x for x in range(n_samples)]

                sess.run(tf.global_variables_initializer())

                for i in range(self.epoch):
                    loss_cur = 0.0
                    shuffle(items)
                    for offset in range(0, n_samples, self.batch_size):
                        items_slice = items[offset:offset + self.batch_size]
                        x_batch, y_batch = self.x_train[items_slice], self.y_train[items_slice]
                        _, loss_val = sess.run([ops, loss],
                                               feed_dict={
                                                   self.x_tensor: x_batch,
                                                   self.y_tensor: y_batch,
                                                   self.keep_prob_tensor: self.train_keep_prob})
                        loss_cur += loss_val

                    print("Epoch: {}, loss: {}".format(i + 1, loss_cur))

                    accurate_validate = self.evaluation(
                        x_data=self.x_validation,
                        y_data=self.y_validation,
                        accuracy_ops=accuracy_ops)
                    print("Validation set accuracy: {}".format(accurate_validate))

                    accurate_train = self.evaluation(
                        x_data=self.x_train,
                        y_data=self.y_train,
                        accuracy_ops=accuracy_ops)
                    print("Training set accuracy: {}".format(accurate_train))
                    print()

                    validation_y.append(accurate_validate)
                    train_y.append(accurate_train)
                    validation_x.append(i + 1)

                # get test accuracy
                loss_test = self.evaluation(
                    x_data=self.x_test,
                    y_data=self.y_test,
                    accuracy_ops=accuracy_ops)
                print("Test set accuracy: {}".format(loss_test))

                print('Training validation set accuracy graph:')
                ax = plt.figure().gca()
                ax.plot(validation_x, validation_y)
                ax.plot(validation_x, train_y)
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                plt.show()
