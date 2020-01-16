import tensorflow as tf
import numpy as np
import ninh
from wheel import *


class Clf:
    def __init__(self, args):
        self.args = args
        self.in_images = tf.placeholder(
            "float32", [None, 224, 224, 3], name="input_images")
        self.in_labels = tf.placeholder(
            "int32", [None, args.n_class], name="input_labels")
        self.in_epsilon = tf.placeholder("float32", [], "epsilon")

        _, _1, _2, self.logit = self._forward(self.in_images)
        self.loss_clf, self.loss_reg = self._add_loss()
        self.accuracy = self._accuracy()
        # self.global_step = tf.Variable(0, trainable=False, name="global_step")
        # self.lr = tf.train.exponential_decay(args.base_lr,
        #                                      self.global_step,
        #                                      args.decay_step,
        #                                      args.decay_rate,
        #                                      staircase=True,
        #                                      name="learning_rate")
        # optim_w = tf.train.MomentumOptimizer(self.lr, args.momentum)
        # optim_b = tf.train.MomentumOptimizer(self.lr * 2.0, args.momentum)
        # var_list_w = [v for v in tf.trainable_variables() if 'weight' in v.name]
        # var_list_b = [v for v in tf.trainable_variables() if 'bias' in v.name]
        # train_w = optim_w.minimize(self.loss_clf + self.loss_reg,
        #                            var_list=var_list_w)
        # train_b = optim_b.minimize(self.loss_clf + self.loss_reg,
        #                            var_list=var_list_b)
        # self.train_op = tf.group(train_w, train_b)
        self.optimizer = tf.train.AdamOptimizer(args.base_lr,
                                                beta1=args.momentum)
        self.train_op = self.optimizer.minimize(self.loss_clf + self.loss_reg)
        self.merged = tf.summary.merge_all()

    def _forward(self, inputs, reuse=False):
        feat = ninh.NINH(inputs, self.args, reuse=reuse)
        # hash_con = ninh.DivideEncode(feat, self.args, self.in_epsilon, reuse=reuse)
        # hash_bin = tf.sign(hash_con - 0.5, "binary_hash")

        logit = fc(feat, "logit", feat.shape.as_list()[-1], self.args.n_class)#, tf.nn.softmax)

        # return feat, hash_con, hash_bin, logit
        return None, None, None, logit

    def _add_loss(self):
        # cross entropy
        with tf.name_scope("xent"):
            loss_xent = tf.nn.softmax_cross_entropy_with_logits(labels=self.in_labels,
                                                                logits=self.logit)
            loss_xent = tf.reduce_sum(loss_xent, axis=-1)
            loss_clf = tf.reduce_mean(loss_xent)

        # weight decay
        var_list = [v for v in tf.trainable_variables()]
        with tf.name_scope("weight_decay"):
            loss_reg = self.args.weight_decay * tf.reduce_mean([tf.nn.l2_loss(x)
                                                           for x in var_list if 'weight' in x.name])

        tf.summary.scalar("loss_clf", loss_clf)
        tf.summary.scalar("loss_reg", loss_reg)
        return loss_clf, loss_reg

    def _accuracy(self):
        acc = tf.reduce_mean(tf.to_float(tf.equal(
            tf.argmax(self.logit, axis=1),
            tf.argmax(self.in_labels, axis=1)
        )))
        tf.summary.scalar("accuracy", acc)
        return acc

    def train_one_step(self, sess, images, labels, epsilon):
        _, summary, l_clf, l_reg, acc = sess.run(
            [self.train_op, self.merged,
                self.loss_clf, self.loss_reg, self.accuracy],
            feed_dict={self.in_images: images,
                       self.in_labels: labels,
                       self.in_epsilon: epsilon})
        return summary, l_clf, l_reg, acc
