# Copyright Charlie Lehman and Martin Halicek. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sonnet as snt
import tensorflow as tf
import numpy as np
import scipy.io as sio
import os
import re
from isic_input import image_batch
from six.moves import xrange
from tqdm import tqdm
import random
from datetime import datetime

BATCH_SIZE = 10
EVAL_SIZE = 1136
NUM_CLASSES = 2
CHECKPOINT_DIR = '/home/charlie/experiments/tf/isic_mbi/'
CHECKPOINT_INTERVAL = 100
MAX_STEPS = 10000
REPORT_INTERVAL = 1
RGB_REDUCE_LEARNING_RATE_INTERVAL = 1000
FFT_REDUCE_LEARNING_RATE_INTERVAL = 500
HSV_REDUCE_LEARNING_RATE_INTERVAL = 1000
RGB_LEARNING_RATE = 1e-2
FFT_LEARNING_RATE = 1e-4
HSV_LEARNING_RATE = 1e-2
LEARNING_RATE_MULTIPLIER = 0.95
NUM_GPU = 2

class CIFAR_NET(snt.AbstractModule):
    """Sonnet implementation of the TensorFlow CIFAR-10 Network
    """
    def __init__(self, name="CIFAR_NET"):
        super(CIFAR_NET, self).__init__(name=name)

    def _build(self, inputs):
        # Initialize all copies of the model the same
        tf.set_random_seed(2017)
        # conv1
        conv1_init= {"w": tf.truncated_normal_initializer(stddev=5e-2),
                    "b": tf.constant_initializer(0.0)}

        conv1_regu = {"w": tf.contrib.layers.l2_regularizer(scale=0.1)}

        outputs = snt.Conv2D(64, [5,5], stride=1,
                            initializers=conv1_init,
                            regularizers=conv1_regu,
                            name="conv1")(inputs)

        outputs = tf.nn.relu(outputs)

        # pool1
        outputs = tf.nn.max_pool(outputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool1')
        # norm1
        outputs = tf.nn.lrn(outputs, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                            name='norm1')
        # conv2
        conv2_init= {"w": tf.truncated_normal_initializer(stddev=5e-2),
                    "b": tf.constant_initializer(0.1)}

        conv2_regu = {"w": tf.contrib.layers.l2_regularizer(scale=0.1)}

        outputs = snt.Conv2D(64, [5,5], stride=1,
                            initializers=conv2_init,
                            regularizers=conv2_regu,
                            name="conv2")(outputs)

        outputs = tf.nn.relu(outputs)

        # norm2
        outputs = tf.nn.lrn(outputs, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                            name='norm2')
        # pool2
        outputs = tf.nn.max_pool(outputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                            padding='SAME', name='pool2')
        # local3
        outputs = snt.BatchFlatten()(outputs)

        local3_init = {"w": tf.truncated_normal_initializer(stddev=0.04),
                       "b": tf.constant_initializer(0.1)}

        local3_regu = {"w": tf.contrib.layers.l2_regularizer(scale=0.004)}

        outputs = snt.Linear(output_size=384,
                            initializers=local3_init,
                            regularizers=local3_regu,
                            )(outputs)

        outputs = tf.nn.relu(outputs)

        # local4
        local4_init = {"w": tf.truncated_normal_initializer(stddev=0.04),
                       "b": tf.constant_initializer(0.1)}

        local4_regu = {"w": tf.contrib.layers.l2_regularizer(scale=0.004)}

        outputs = snt.Linear(output_size=192,
                            initializers=local4_init,
                            regularizers=local4_regu,
                            )(outputs)
        outputs = tf.nn.relu(outputs)

        # linear layer (WX + b)
        linear_init = {"w": tf.truncated_normal_initializer(stddev=1 / 192.0),
                       "b": tf.constant_initializer(0.0)}

        linear_regu = {"w": tf.contrib.layers.l2_regularizer(scale=0.0)}

        outputs = snt.Linear(output_size=NUM_CLASSES,
                            initializers=linear_init,
                            regularizers=linear_regu,
                            )(outputs)
        return outputs


def _add_loss_summaries(total_loss):
    model_name = re.sub('/total_loss', '', total_loss.op.name)
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('%s/losses' % model_name)
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar('avg_train/%s' % l.op.name, loss_averages.average(l))
        tf.summary.scalar('train/%s' % l.op.name, l)
    return loss_averages_op

def loss(logits, labels):
    model_name = re.sub('/linear_2/add', '', logits.op.name)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('%s/losses' % model_name, loss)
    return tf.add_n(tf.get_collection('%s/losses' % model_name), name='%s' % model_name)

def tower_loss(scope, rgb_inf, fft_inf, hsv_inf):
    rgb_images, fft_images, hsv_images, labels = image_batch(BATCH_SIZE)
    model_logits = [rgb_inf(rgb_images), fft_inf(fft_images), hsv_inf(hsv_images)]
    model_names = [re.sub('/linear_2/add', '', logits.op.name) for logits in model_logits]
    model_total_losses = []
    for logits, name in zip(model_logits, model_names):
        _ = loss(logits, labels)
        losses = tf.get_collection('%s/losses' % name, scope)
        total_loss = tf.add_n(losses, name='total_loss')
        model_total_loss.append(total_loss)
        for l in losses + [total_loss]:
            loss_name = re.sub('%s_[0-9]*/'% TOWER_NAME, '', l.op.name)
            tf.summary.scalar(loss_name, l)

    return model_total_losses

def average_gradients(tower_grads):
    # transpose the list of lists
    tower_grads = list(map(list, zip(*tower_grads)))
def mutli_gpu_train():
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        rgb_inf = CIFAR_NET(name='rgb_net')
        fft_inf = CIFAR_NET(name='fft_net')
        hsv_inf = CIFAR_NET(name='hsv_net')

        trainable_variables = tf.trainable_variables()

        rgb_lr = tf.train.exponential_decay(RGB_LEARNING_RATE,
                                            global_step,
                                            reduce_learning_rate_interval,
                                            LEARNING_RATE_MULTIPLIER,
                                            staircase=True)

        fft_lr = tf.train.exponential_decay(HSV_LEARNING_RATE,
                                            global_step,
                                            reduce_learning_rate_interval,
                                            LEARNING_RATE_MULTIPLIER,
                                            staircase=True)

        hsv_lr = tf.train.exponential_decay(FFT_LEARNING_RATE,
                                            global_step,
                                            reduce_learning_rate_interval,
                                            LEARNING_RATE_MULTIPLIER,
                                            staircase=True)

        rgb_optimizer = tf.train.GradientDescentOptimizer(rgb_lr)
        fft_optimizer = tf.train.GradientDescentOptimizer(fft_lr)
        hsv_optimizer = tf.train.GradientDescentOptimizer(hsv_lr)


        rgb_train_step = rgb_optimizer.apply_gradients(rgb_grads, global_step=global_step)
        fft_train_step = fft_optimizer.apply_gradients(fft_grads, global_step=None)
        hsv_train_step = hsv_optimizer.apply_gradients(hsv_grads, global_step=None)
        tower_grads = []
        for i in xrange(NUM_GPU):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d', (TOWER_NAME, i)) as scope:
                    rgb_loss, fft_loss, hsv_loss = tower_loss(scope, rgb_inf, fft_inf, hsv_inf)
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    rgb_grads = rgb_optimizer.compute_gradients(rgb_loss)
                    fft_grads = fft_optimizer.compute_gradients(fft_loss)
                    hsv_grads = hsv_optimizer.compute_gradients(hsv_loss)

                    tower_grads.append([rgb_grads, fft_grads, hsv_grads])


def train(num_training_iterations, report_interval, with_test=False):
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        rgb_inf = CIFAR_NET(name='rgb_net')
        fft_inf = CIFAR_NET(name='fft_net')
        hsv_inf = CIFAR_NET(name='hsv_net')
        rgb_images, fft_images, hsv_images, labels = image_batch(BATCH_SIZE)

        if with_test:
            test_rgb_images, test_fft_images, test_hsv_images, test_labels = image_batch(
                BATCH_SIZE, train=False)
            test_rgb_logits = rgb_inf(test_rgb_images)
            test_fft_logits = fft_inf(test_fft_images)
            test_hsv_logits = hsv_inf(test_hsv_images)

            tf.summary.scalar('test/rgb', loss(test_rgb_logits, test_labels))
            tf.summary.scalar('test/fft', loss(test_fft_logits, test_labels))
            tf.summary.scalar('test/hsv', loss(test_hsv_logits, test_labels))


        tf.summary.image('images/rgb', rgb_images, max_outputs=1)
        tf.summary.image('images/fft', fft_images, max_outputs=1)
        tf.summary.image('images/hsv', hsv_images, max_outputs=1)

        rgb_logits = rgb_inf(rgb_images)
        fft_logits = fft_inf(fft_images)
        hsv_logits = hsv_inf(hsv_images)

        names = {'rgb+fft':tf.constant([1,1,0], dtype=tf.float32),
                 'rgb+hsv':tf.constant([1,0,1], dtype=tf.float32),
                 'fft+hsv':tf.constant([0,1,1], dtype=tf.float32),
                 'all':tf.constant([1,1,1], dtype=tf.float32)}

        majority_vote_fuse(rgb_logits, fft_logits, hsv_logits, labels, names)

        rgb_loss = loss(rgb_logits, labels)
        fft_loss = loss(fft_logits, labels)
        hsv_loss = loss(hsv_logits, labels)

        trainable_variables = tf.trainable_variables()

        rgb_lr = tf.train.exponential_decay(RGB_LEARNING_RATE,
                                            global_step,
                                            RGB_REDUCE_LEARNING_RATE_INTERVAL,
                                            LEARNING_RATE_MULTIPLIER,
                                            staircase=True)

        fft_lr = tf.train.exponential_decay(FFT_LEARNING_RATE,
                                            global_step,
                                            FFT_REDUCE_LEARNING_RATE_INTERVAL,
                                            LEARNING_RATE_MULTIPLIER,
                                            staircase=True)

        hsv_lr = tf.train.exponential_decay(HSV_LEARNING_RATE,
                                            global_step,
                                            HSV_REDUCE_LEARNING_RATE_INTERVAL,
                                            LEARNING_RATE_MULTIPLIER,
                                            staircase=True)

        rgb_loss_avg = _add_loss_summaries(rgb_loss)
        fft_loss_avg = _add_loss_summaries(fft_loss)
        hsv_loss_avg = _add_loss_summaries(hsv_loss)

        with tf.control_dependencies([rgb_loss_avg, fft_loss_avg, hsv_loss_avg]):
            rgb_optimizer = tf.train.GradientDescentOptimizer(rgb_lr)
            fft_optimizer = tf.train.GradientDescentOptimizer(fft_lr)
            hsv_optimizer = tf.train.GradientDescentOptimizer(hsv_lr)

            rgb_grads = rgb_optimizer.compute_gradients(rgb_loss)
            fft_grads = fft_optimizer.compute_gradients(fft_loss)
            hsv_grads = hsv_optimizer.compute_gradients(hsv_loss)

        rgb_train_step = rgb_optimizer.apply_gradients(rgb_grads, global_step=global_step)
        fft_train_step = fft_optimizer.apply_gradients(fft_grads, global_step=None)
        hsv_train_step = hsv_optimizer.apply_gradients(hsv_grads, global_step=None)

        saver = tf.train.Saver()

        checkpoint_hooks = [tf.train.CheckpointSaverHook(
                            checkpoint_dir=CHECKPOINT_DIR,
                            save_steps=CHECKPOINT_INTERVAL,
                            saver=saver)]

        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(CHECKPOINT_DIR)

        with tf.train.SingularMonitoredSession(hooks=checkpoint_hooks, checkpoint_dir=CHECKPOINT_DIR) as sess:
            start_iteration = sess.run(global_step)
            for train_iteration in xrange(start_iteration, num_training_iterations):
                rgb_loss_, fft_loss_, hsv_loss_, rgb_train_step_, fft_train_step_, hsv_train_step_, sum_str  = sess.run(
                        [rgb_loss, fft_loss, hsv_loss, rgb_train_step, fft_train_step, hsv_train_step, summary_op])


                if train_iteration % report_interval == 0 and not train_iteration == 0:
                    print("%d: Training Loss (RGB:%.3f, FFT:%.3f, HSV:%.3f)" % (train_iteration, rgb_loss_, fft_loss_, hsv_loss_))
                    summary_writer.add_summary(sum_str, train_iteration)

def majority_vote_fuse(rgb_logits, fft_logits, hsv_logits, labels, names, evaluate=False):
    scores = {}
    for name, mask in names.items():
        logits = (tf.round(tf.nn.softmax(tf.scalar_mul(mask[0], rgb_logits)))+
                  tf.round(tf.nn.softmax(tf.scalar_mul(mask[1], fft_logits)))+
                  tf.round(tf.nn.softmax(tf.scalar_mul(mask[2], hsv_logits))))
        if evaluate:
            tp, fp, tn, fn = binary_score(logits, labels)
            scores[name] = {}
            scores[name]['tp'] = tp
            scores[name]['fp'] = fp
            scores[name]['tn'] = tn
            scores[name]['fn'] = fn
        else:
            tf.summary.scalar('fuse/%s' % name, loss(logits, labels))
    if evaluate:
        return scores
    else:
        return

def evaluate():
    rgb_images, fft_images, hsv_images, labels = image_batch(train=False, batch_size=BATCH_SIZE)
    rgb_inf = CIFAR_NET(name='rgb_net')
    fft_inf = CIFAR_NET(name='fft_net')
    hsv_inf = CIFAR_NET(name='hsv_net')
    rgb_logits = rgb_inf(rgb_images)
    fft_logits = fft_inf(fft_images)
    hsv_logits = hsv_inf(hsv_images)
    names = {'rgb':tf.constant([1,0,0], dtype=tf.float32),
             'fft':tf.constant([0,1,0], dtype=tf.float32),
             'hsv':tf.constant([0,0,1], dtype=tf.float32),
             'rgb+fft':tf.constant([1,1,0], dtype=tf.float32),
             'rgb+hsv':tf.constant([1,0,1], dtype=tf.float32),
             'fft+hsv':tf.constant([0,1,1], dtype=tf.float32),
             'all':tf.constant([1,1,1], dtype=tf.float32)}
    scores = majority_vote_fuse(rgb_logits, fft_logits, hsv_logits, labels, names, evaluate=True)


    with tf.train.SingularMonitoredSession(checkpoint_dir=CHECKPOINT_DIR) as sess:

        scores_c = {}
        predictions = []
        step = 0
        names_ = sess.run(names)
        for name in names_.keys():
            scores_c[name] = {}
            scores_c[name]['tp_c'] = 0
            scores_c[name]['fp_c'] = 0
            scores_c[name]['tn_c'] = 0
            scores_c[name]['fn_c'] = 0
        print('Compiling Scores')
        for _ in tqdm(range(EVAL_SIZE//BATCH_SIZE)):
            scores_  = sess.run(scores)
            for name in scores_.keys():
                scores_c[name]['tp_c'] += np.sum(scores_[name]['tp'])
                scores_c[name]['fp_c'] += np.sum(scores_[name]['fp'])
                scores_c[name]['tn_c'] += np.sum(scores_[name]['tn'])
                scores_c[name]['fn_c'] += np.sum(scores_[name]['fn'])
            step += BATCH_SIZE

        for name, score_c in scores_c.items():
            tp_c = score_c['tp_c']
            fp_c = score_c['fp_c']
            tn_c = score_c['tn_c']
            fn_c = score_c['fn_c']
            accuracy = (tn_c+tp_c) / (tp_c+fp_c+tn_c+fn_c)
            sensitivity = tp_c/(tp_c+fn_c)
            specificity = tn_c/(tn_c+fp_c)

            print('====================')
            print('%s' % (name))
            print('====================')
            print('accuracy = %.3f' % (accuracy))
            print('sensitivity = %.3f' % (sensitivity))
            print('specificity = %.3f' % (specificity))
            print('True Pos = %i' %  (tp_c))
            print('False Pos = %i' % (fp_c))
            print('True Neg = %i' %  (tn_c))
            print('False Neg = %i' % (fn_c))

def binary_score(logits,labels):
    is_label_one = tf.cast(labels, dtype=tf.bool)
    is_label_zero = tf.logical_not(is_label_one)
    correct_prediction = tf.nn.in_top_k(logits, labels, 1, name="correct_answers")
    false_prediction = tf.logical_not(correct_prediction)
    true_positives = tf.reduce_sum(tf.to_int32(tf.logical_and(correct_prediction,is_label_one)))
    false_positives = tf.reduce_sum(tf.to_int32(tf.logical_and(false_prediction, is_label_zero)))
    true_negatives = tf.reduce_sum(tf.to_int32(tf.logical_and(correct_prediction, is_label_zero)))
    false_negatives = tf.reduce_sum(tf.to_int32(tf.logical_and(false_prediction, is_label_one)))

    return true_positives, false_positives, true_negatives, false_negatives
def main(argv = None):
    train(MAX_STEPS, REPORT_INTERVAL, with_test=True)
    evaluate()

if __name__ == "__main__":
    tf.app.run()
