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

from tqdm import tqdm
import os
import random
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
import tensorflow as tf

IMAGE_SHAPE = [220, 220, 3]
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 4540
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 1136
PATH_TO_IMAGES = 'isic_cnn_data'
TRAIN_NAME = 'isic_train'
TEST_NAME = 'isic_test'

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def shuffled_file_list_with_labels(data_dir):
    if not os.path.isdir(data_dir):
        print('Did you download the dataset?:')
        print('==============================')
        print('PATH_TO_IMAGES = \'%s\'' % PATH_TO_IMAGES)
        raise FileNotFoundError

    list_of_files = []
    for path, subdirs, files in os.walk(data_dir):
        for name in files:
            list_of_files += [os.path.join(path,name)]

    random.shuffle(list_of_files)
    labels = [int('malignant' in f) for f in list_of_files]
    return list_of_files, labels

def load_images(list_of_files, size):
    if len(list_of_files) == 0:
        print('There seems to be nothing in \'%s\'' % PATH_TO_IMAGES)
        print('=========================================')
        raise FileNotFoundError
    images = []
    print('Loading ALL images into memory!')
    for f in tqdm(list_of_files):
        images.append(imresize(imread(f), size))
    print('Loaded %i' % len(images))
    return images

def make_tfrecord(images, labels, idx, name):
    writer = tf.python_io.TFRecordWriter('%s.tfrecords' % name)
    print('Building TFRecord named %s.tfrecords' % name)
    for i in tqdm(idx):
        image = images[i] / 256.0
        image_bytes = image.tostring()
        label= int(labels[i])
        example = tf.train.Example(
                features = tf.train.Features(
                    feature={
                        'image_bytes': _bytes_feature(tf.compat.as_bytes(image_bytes)),
                        'label': _int64_feature(label)
                        }))
        writer.write(example.SerializeToString())
    print('Saving the record')


def read_and_decode_single_example(filename):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([],tf.int64),
                'image_bytes': tf.FixedLenFeature([],tf.string),
                })


    rgb_im = tf.decode_raw(features['image_bytes'], tf.float64)
    rgb_im = tf.image.convert_image_dtype(rgb_im, tf.float32)
    rgb_im = tf.reshape(rgb_im, IMAGE_SHAPE)
    rgb_im.set_shape(IMAGE_SHAPE)

    label = tf.cast(features['label'], tf.int64) 

    return rgb_im, label

def image_batch(batch_size, train=True):
    min_fraction_of_examples_in_queue = 0.4
    with tf.device('/cpu:0'):
        if train:
            image, label = read_and_decode_single_example('%s.tfrecords' % TRAIN_NAME)
            image = tf.random_crop(tf.image.resize_images(image, np.multiply(IMAGE_SHAPE[:-1],2)), IMAGE_SHAPE)
            min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                min_fraction_of_examples_in_queue)

        else:
            image, label = read_and_decode_single_example('%s.tfrecords' % TEST_NAME)
            min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TEST *
                                min_fraction_of_examples_in_queue)

        rgb_im, fft_im, hsv_im = rgb_fft_hsv(image)

    with tf.device('/cpu:0'):
        rgb_images, fft_images, hsv_images, labels = tf.train.shuffle_batch(
            [rgb_im, fft_im, hsv_im, label],
            batch_size=batch_size,
            num_threads=8,
            capacity=min_queue_examples+3*batch_size,
            min_after_dequeue=min_queue_examples)
    return rgb_images, fft_images, hsv_images, labels

def _fft2d_shift(image):
    indexes = len(image.get_shape())-1
    top, bottom = tf.split(image, 2, indexes)
    image = tf.concat([bottom, top], axis=indexes)
    left, right = tf.split(image, 2, indexes-1)
    image = tf.concat([right, left], axis=indexes - 1)
    return image

def _fft_convert(image):
    H, W, C = image.get_shape().as_list()
    imstack = []
    image = tf.log(tf.abs(tf.spectral.rfft3d(image, fft_length=[H,W,C+1])))
    for i in range(image.get_shape().as_list()[-1]):
        imstack.append(_fft2d_shift(image[:,:,i]))
    image = tf.stack(imstack, axis=2)
    return image

def rgb_fft_hsv(image):
    hsv_im = tf.image.rgb_to_hsv(image)
    with tf.device('/gpu:0'):
        fft_im = _fft_convert(image)
    rgb_im = preprocess(image)
    hsv_im = preprocess(hsv_im)
    fft_im = preprocess(fft_im)
    return rgb_im, fft_im, hsv_im

def preprocess(image):
    image = tf.image.per_image_standardization(image)
    return image

def main(argv = None):
    im_list, labels = shuffled_file_list_with_labels(PATH_TO_IMAGES)
    images = load_images(im_list, IMAGE_SHAPE)
    idx = np.arange(0, len(images))
    np.random.shuffle(idx)
    slice_point = int(len(idx)//1.25)
    train = idx[:slice_point]
    test = idx[slice_point:]
    make_tfrecord(images, labels, train, TRAIN_NAME )
    make_tfrecord(images, labels, test, TEST_NAME)

if __name__ == "__main__":
    tf.app.run()
