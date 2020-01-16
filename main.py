import argparse
import os
import time
import tensorflow as tf
import numpy as np
from scipy.ndimage import zoom
import models
from tensorflow.examples.tutorials.mnist import input_data


parser = argparse.ArgumentParser(description='NINH')
parser.add_argument('--gpu_id', type=str, nargs='?', default="0")
parser.add_argument('--gpu_frac', type=float, default=0.5,
                    help="fraction of gpu memory to use")
parser.add_argument('--log_path', type=str, default="log")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_class', type=int, default=10, help="#classes")
parser.add_argument('--bit', type=int, default=32, help="#bit")
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--epsilon', type=float, default=0.5,
                    help="for devide-and-encode layer")
parser.add_argument('--base_lr', type=float, default=1e-4,
                    help="base learning rate")
parser.add_argument('--momentum', type=float, default=0.9, help="momentum")
parser.add_argument('--decay_step', type=int, default=300)
parser.add_argument('--decay_rate', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float,
                    default=0.0005, help="weight decay")
parser.add_argument('--max_iter', type=int, default=3000, help="max #iteration")
parser.add_argument('--test_per', type=int, default=50, help="test interval")
args = parser.parse_args()


def timestamp():
    """time-stamp string: Y-M-D-h-m"""
    t = time.localtime(time.time())
    return "{}-{}-{}-{}-{}".format(
        t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)


if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)
log_file_path = os.path.join(args.log_path, "log.{}".format(timestamp()))
log_file = open(log_file_path, "a")


def transform(img):
    """[None, 784]
    -> [None, 28, 28, 1]
    -> [None, 28, 28, 3]
    -> [None, 224, 224, 3]
    """
    # print(img.shape)
    img = img.reshape(-1, 28, 28, 1)
    img = np.repeat(img, 3, axis=3)
    return zoom(img, [1, 8, 8, 1], order=0)


def test(sess, model, dataset):
    n_total = dataset.test.num_examples  # 10k
    batch = 100
    acc = 0
    for i in range(n_total // batch):
        image, label = dataset.test.next_batch(batch)
        image = transform(image)
        acc += sess.run(model.accuracy,
                        feed_dict={model.in_images: image,
                                   model.in_labels: label,
                                   model.in_epsilon: 0.4})

    acc = acc * batch / n_total
    return acc


def train(sess, tf_writer, model, dataset):
    log_file.write("begin time: {}\n".format(time.asctime()))

    for epoch in range(args.max_iter):
        image, label = dataset.train.next_batch(args.batch_size)
        image = transform(image)
        summary, l_clf, l_reg, acc = model.train_one_step(
            sess, image, label, 0.4)
        tf_writer.add_summary(summary, epoch)

        if epoch % args.test_per == 0:
            log_file.write("--- iter: {}\n".format(epoch))
            acc = test(sess, model, dataset)
            print("epoch:", epoch, ", acc:", acc)
            log_file.write("acc: {}\n".format(acc))

    log_file.write("end time: {}\n".format(time.asctime()))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    gpu_cfg = tf.ConfigProto()
    # gpu_cfg.gpu_options.per_process_gpu_memory_fraction = args.gpu_frac
    # gpu_cfg.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_cfg)

    dataset = input_data.read_data_sets(
        "/usr/local/dataset/MNIST/", one_hot=True)
    model = models.Clf(args)

    tf_writer = tf.summary.FileWriter(args.log_path, sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    os.system("clear")
    train(sess, tf_writer, model, dataset)

    sess.close()

log_file.close()
