#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   Project name: RetinaNet-Keras
#   File name   : tmp.py
#   Author      : ylqi007
#   Created date: 2021-01-01 7:24 PM
#
# ================================================================
"""
keras.utils.get_file(): https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file
ZipFile Objects: https://docs.python.org/3/library/zipfile.html#zipfile-objects
TensorFlow Datasets: https://www.tensorflow.org/datasets/overview
    TensorFlow Datasets: a collection of ready-to-use datasets.
    TFDS is different with tf.data
"""

import os
import re
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

"""
## Downloading the COCO2017 dataset

Training on the entire COCO2017 dataset which has around 118k images takes a lot of time, 
hence we will be using a smaller subset of ~500 images for training in this example.
"""

# url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
# filename = os.path.join(os.getcwd(), "data.zip")
# keras.utils.get_file(filename, url)
#
# with zipfile.ZipFile("data.zip", "r") as z_fp:
#     z_fp.extractall("./")



"""
## Implementing utility functions
Bounding boxes can be represented in multiple ways, the most common formats are:
- Storing the coordinates of the corners `[xmin, ymin, xmax, ymax]`
- Storing the coordinates of the center and the box dimensions
`[x, y, width, height]`
Since we require both formats, we will be implementing functions for converting
between the formats.
"""


def swap_xy(boxes):
    """Swaps order the of x and y coordinates of the boxes.
    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.
    Returns:
      swapped boxes with shape same as that of boxes.
    """
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def convert_to_xywh(boxes):
    """ Changes the box format to center, width and height, i.e. [x, y, w, h]

    :param boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)` representing
        bounding boxes where each box is of the format `[xmin, ymin, xmax, ymax]`.

    :return: Converted boxes with shape as that of boxes. `[x, y, width, height]`
    """
    # boxes is with shape (num_boxes, 4), boxes[:, 0] is with shape (num_boxes,)
    # tf.concat([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]]), shape=(None, 2)
    # tf.concat([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1), shape=(None, 4)
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


def convert_to_corners(boxes):
    """Changes the box format to corner coordinates
    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.
    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )


"""
## Preprocessing data

Preprocessing the images involves two steps(resize + augmentation):
- Resizing the image: Images are resized such that the shortest size is equal to 800 px, 
after resizing if the longest side of the image exceeds 1333 px, the image is resized such that 
the longest size is now capped at 1333 px. 
- Applying augmentation: Random scale jittering and random horizontal flipping are the only 
augmentations applied to the images. Along with the images, bounding boxes are rescaled and 
flipped if required.
"""


def random_flip_horizontal(image, boxes):
    """ Flips image and boxes horizontally with 50% chance.

    :param image: A 3-D tensor of shape '(height, width, channels)' representing an image.
    :param boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.
        i.e. original: [xmin, ymin, xmax, ymax]
        flipped: [1-xmax, ymin, 1-xmin, ymax]

    :return: Randomly flipped image and boxes.
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack([1 - boxes[:, 2], boxes[:, 1],
                          1 - boxes[:, 0], boxes[:, 3]], axis=-1)
    return image, boxes


def resize_and_pad_image(image,
                         min_side=800.0, max_side=1333.0,
                         jitter=[640, 1024],
                         stride=128.0):
    """ Resizes and pads image while preserving aspect ratio. Make sure that the original image
    and rescaled image have the same top left corner.

    1. Resize images so that the shorter side is equal to `min_side`
    2. If the longer side is greater than `max_side`, then resize the image with longer side
        equal to `max_side`.
    3. Pad with zeros on right and bottom to make the image shape divisible by `stride`.

    :param image: A 3-D tensor of shape `(height, width, channels)` representing an image.
    :param min_side: The shorter side of the image is resized to this value, if `jitter` is set
        to None.
    :param max_side: If the longer side of the image exceeds this value after resizing, the image
        is resized such that the longer side now equals to this value.
    :param jitter: A list of floats containing minimum and maximum size for scale jittering. If
        available, the shorter side of the image will be resized to a random value in this range.
    :param strides: The stride for the smallest feature map in the feature pyramid. Can be
        calculated using `image_size / feature_map_size`.
    :return:
        image: Resized and padded image.
        image_shape: Shape of the image before padding.
        ratio: The scaling factor used to resize the image.
    """
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)    # height, width

    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)

    image_shape = ratio * image_shape   # image shape after resizing
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))    # resize image
    padded_image_shape = tf.cast(tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32)
    image = tf.image.pad_to_bounding_box(image, 0, 0,
                                         padded_image_shape[0], padded_image_shape[1])
    return image, image_shape, ratio


def preprocess_data(sample):
    """ Applies preprocessing step to a single sample.

    :param sample: A dict representing a single training sample. The element in tf.data.Dataset
        is dict.
        * The original sample['objects']['bbox'] is with shape (None, 4), and the format of each
        element is [ymin, xmin, ymax, xmax], dtype=float32
        reference: https://github.com/tensorflow/datasets/blob/9b5e431f1c9d90e207d69972edc1d71ee02b57e4/tensorflow_datasets/object_detection/coco.py#L387
        * The original sample['objects']['label'] is with shape (None,), dtype=int64

    :return:
        image: Resized and padded image with random horizontal flipping applied.
        bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is of the format
            `[x, y, width, height]`.
        class_id: An tensor representing the class id of the objects, having shape `(num_objects,)`
    """
    image = sample['image']
    bbox = swap_xy(sample['objects']['bbox'])
    class_id = tf.cast(sample['objects']['label'], dtype=tf.int32)  # Cast from int64 to int32

    # Resize image
    image, bbox = random_flip_horizontal(image, bbox)       # Random flip image and bboxes
    image, image_shape, _ = resize_and_pad_image(image)

    # Resize bboxes
    # image_shape[0]: height, image_shape[1]: width
    # bbox is normalized format. bbox[:, 0] * image_shape[1] will be not normalized.
    bbox = tf.stack([bbox[:, 0] * image_shape[1],
                     bbox[:, 1] * image_shape[0],
                     bbox[:, 2] * image_shape[1],
                     bbox[:, 3] * image_shape[0]], axis=-1)

    # Convert [xmin, ymin, xmax, ymax] to [x, y, width, height]
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id


"""
## Load the COCO2017 dataset using TensorFlow Datasets

- coco/2017: https://www.tensorflow.org/datasets/catalog/coco#coco2017

* Set `data_dir=None` to load the complete dataset
* Load a dataset: https://www.tensorflow.org/datasets/overview#load_a_dataset
* TensorFlow Datasets 数据集载入: https://tf.wiki/zh_hans/appendix/tfds.html#tensorflow-datasets
"""
# Download the complete COCO2017 dataset as tfrecords
# (train_dataset, val_dataset), dataset_info = tfds.load(
#     "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
# )

data_dir = "/home/ylqi007/work/DATA/COCO"
(train_dataset, val_dataset), dataset_info = tfds.load(
    "coco/2017", split=["train", "validation"], with_info=True, data_dir=data_dir)

# print(train_dataset)
# print(val_dataset)
# print(dataset_info)


"""
## Setting up a `tf.data` pipeline

To ensure that the model is fed with data efficiently we will be using `tf.data` API to create 
our input pipeline. The input pipeline consists for the following major processing steps:
- Apply the preprocessing function to the samples
- Create batches with fixed batch size. Since images in the batch can have different dimensions, 
and can also have different number of objects, we use `padded_batch` to the add the necessary  
padding to create rectangular tensors
- Create targets for each sample in the batch using `LabelEncoder`

* tf.data.experimental.AUTOTUNE: Parallelizing data transformation, https://www.tensorflow.org/guide/data_performance#parallelizing_data_transformation
"""
autotune = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
# train_dataset = train_dataset.shuffle(8 * batch_size)
# train_dataset = train_dataset.padded_batch(
#     batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
# )
# train_dataset = train_dataset.map(
#     label_encoder.encode_batch, num_parallel_calls=autotune
# )
# train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
# train_dataset = train_dataset.prefetch(autotune)
#
# val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
# val_dataset = val_dataset.padded_batch(
#     batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
# )
# val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
# val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
# val_dataset = val_dataset.prefetch(autotune)

i = 0
for sample in train_dataset:
    if i < 1:
        i += 1
        print('sample[0]:\n', type(sample[0]), sample[0].shape)
        print('sample[1]:\n', type(sample[1]), sample[1])
        print('sample[2]:\n', type(sample[2]), sample[2])
    else:
        break