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
