import tensorflow as tf
from wheel import *


def NINH(x, args, var_scope="NINH", reuse=False):
    with tf.variable_scope(var_scope, reuse=reuse):
        x = conv(x, "conv1", 11, 4, 3, 96, padding="VALID")
        x = conv(x, "conv2", 1, 1, 96, 96)
        x = tf.layers.max_pooling2d(x, 3, 2, padding="SAME", name="pool1")
        print(x.shape.as_list())
        x = conv(x, "conv3", 5, 1, 96, 256, padding="SAME")
        x = conv(x, "conv4", 1 , 1 ,256, 256, padding="SAME")
        x = tf.layers.max_pooling2d(x, 3, 2, name="pool2")
        print(x.shape.as_list())
        x = conv(x, "conv5", 3, 1, 256, 384)
        x = conv(x, "conv6", 1, 1, 384, 384)
        x = tf.layers.max_pooling2d(x, 3, 2, name="pool3")
        print(x.shape.as_list())
        x = conv(x, "conv7", 3, 1, 384, 1024)
        # x = conv(x, "conv8", 1, 1, 1024, 50*args.bit)
        x = conv(x, "conv8", 1, 1, 1024, args.n_class)
        x = tf.layers.average_pooling2d(x, 6, 1, name="pool4")
        print(x.shape.as_list())
        # x = tf.reshape(x, [-1, 50*args.bit])
        x = tf.reshape(x, [-1, args.n_class])
        print(x.shape.as_list())

    return x


def DivideEncode(x, args, epsilon, var_scope="divide_and_encode", reuse=False):
    with tf.variable_scope(var_scope, reuse=reuse):
        x_list = tf.split(x, args.bit, axis=1, name="split")
        for i in range(args.bit):
            tmp = fc(x_list[i], "bit_{}".format(i), 50, 1, act_fn=None)
            tmp = tf.nn.sigmoid(args.beta * tmp)
            tmp = pw_threshold(tmp, epsilon)
            x_list[i] = tmp

        x = tf.concat(x_list, axis=1)

    return x
