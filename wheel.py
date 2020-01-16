import tensorflow as tf


def conv(x, name, filter_size, stride, in_channel, out_channel, padding='SAME', act_fn=tf.nn.relu):
    """conv 2d"""
    with tf.variable_scope(name):
        w = tf.get_variable(
            'weight_'+name, shape=[filter_size, filter_size, in_channel, out_channel])
        b = tf.get_variable('bias_'+name, shape=[out_channel])

        out = tf.nn.conv2d(x, w,
                           strides=[1, stride, stride, 1],
                           padding=padding)
        out = tf.reshape(tf.nn.bias_add(out, b),
                         [-1]+out.get_shape().as_list()[1:])
        if act_fn:
            out = act_fn(out)

    print(name, ':', out.shape.as_list())
    return out


def fc(x, name, num_in, num_out, act_fn=None, stddev=0.01):
    """fully connected layer"""
    with tf.variable_scope(name):
        w = tf.get_variable('weight_'+name, initializer=tf.truncated_normal(
            [num_in, num_out], stddev=stddev))
        b = tf.get_variable(
            'bias_'+name, initializer=tf.constant(0.1, shape=[num_out]))

        out = tf.nn.xw_plus_b(x, w, b, name=name)
        if act_fn:
            out = act_fn(out)

    print(name, ':', out.shape.as_list())
    return out


@tf.custom_gradient
def pw_threshold(x, epsilon):
    """piece-wise threshold"""
    cond_org = ((0.5 - epsilon) <= x) & (x < (0.5 + epsilon))
    cond_one = x >= (0.5 + epsilon)
    ones = tf.ones_like(x)
    zeros = tf.zeros_like(x)
    y = tf.where(cond_org, x, zeros) + \
        tf.where(cond_one, ones, zeros)

    def grad(dy):
        cond = ((0.5 - epsilon) <= x) & (x < (0.5 + epsilon))
        zeros = tf.zeros_like(dy)
        return tf.where(cond, dy, zeros), epsilon

    return y, grad
