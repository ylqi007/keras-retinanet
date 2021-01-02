#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   Project name: RetinaNet-Keras
#   File name   : preprocessing.py
#   Author      : ylqi007
#   Created date: 2021-01-02 12:05 AM
#
# ================================================================

import tensorflow as tf
from utils.utils import *


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

