#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   Project name: RetinaNet-Keras
#   File name   : utils.py
#   Author      : ylqi007
#   Created date: 2020-12-31 11:09 PM
#
# ================================================================


import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt


"""
## Implementing utility functions

Bounding boxes can be represented in multiple ways, the most common formats are:
- Storing the coordinates of the corners `[xmin, ymin, xmax, ymax]`
- Storing the coordinates of the center and the box dimensions `[x, y, width, height]`

Since we require both formats, we will be implementing functions for converting between the formats.
"""


def swap_xy(boxes):
    """
    Swaps order of the x and y coordinates of the boxes.
    For example, [ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax], or vice versa.

    :param boxes: A tensor with shape (num_boxes, 4) representing bounding boxes.

    :return: swapped boxes with shape same as that of boxes.
    """
    # boxes is with shape (num_boxes, 4), boxes[:, 0] is with shape (num_boxes,)
    # boxes[:, i].shape = (num_boxes,), when using `tf.stack([], axis=-1)`, it will reshape boxes[:, i] to (num_boxes, 1), and then stack them together.
    # tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]]), shape=(4, None)
    # tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1), shape=(None, 4)
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def convert_to_xywh(boxes):
    """ Changes the box format to center, width and height,
    i.e. [xmin, ymin, xmax, ymax] to [x, y, w, h]

    :param boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)` representing
        bounding boxes where each box is of the format `[xmin, ymin, xmax, ymax]`.

    :return: Converted boxes with shape as that of boxes. `[x, y, width, height]`
    """
    # boxes is with shape (num_boxes, 4), boxes[:, 0] is with shape (num_boxes,)
    # (boxes[..., :2] + boxes[..., 2:]) / 2.0, shape is (num_boxes, 2)
    # boxes[..., 2:] - boxes[..., :2]], shape is (num_boxes, 2)
    # After tf.concat(..., axis=-1), (num_boxes, 2) + (num_boxes, 2) will become (num_boxes, 2+2)
    # ??? tf.concat([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]]), shape=(None, 2)
    # tf.concat([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1), shape=(None, 4)
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


def convert_to_corners(boxes):
    """ Changes the box format to corner coordinates,
    i.e. [x, y, width, height] to [xmin, ymin, xmax, ymax]

    :param boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    :return: Converted boxes with shape same as that of boxes.
    """
    # [x, y, width, height] ==> [xmin, ymin, xmax, ymax]
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )


# TODO: Here is a split line.


"""
## Computing pairwise Intersection Over Union (IOU)

As we will see later in the example, we would be assigning ground truth boxes to anchor boxes 
based on the extend of overlapping. This will require us to calculate the Intersection Over 
Union (IOU) between all the anchor boxes and ground truth boxes pairs.
将 ground truth box，匹配到 anchor boxes。
"""


def compute_iou(boxes1, boxes2):
    """
    Compute pairwise IoU matrix for given tow sets of boxes.

    :param boxes1: A tensor with shape `(M, 4)` representing bounding boxes where each box is of
        the format `[x, y, width, height]`.
    :param boxes2: A tensor with shape `(N, 4)` representing bounding boxes where each box is of
        the format `[x, y, width, height]`.

    :return: pairwise IoU matrix with shape `(M, N)`, where the values at ith row jth column
        holds the IoU between ith box and jth box from boxes1 and boxes2 respectively.
    """
    boxes1_corners = convert_to_corners(boxes1) # [x, y, width, height] ==> [xmin, ymin, xmax, ymax]
    boxes2_corners = convert_to_corners(boxes2) # [x, y, width, height] ==> [xmin, ymin, xmax, ymax]
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2]) # (M,1,2) - (N,2) => (M,N,2)
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)     # width and height of intersection area, (M, N, 2)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]   # (M, N)
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]   # shape=(M,)
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]   # shape=(N,)
    # boxes1_area[:, None] + boxes2_area ==> (M, N)
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)


# TODO: Here is a split line.


def visualize_detections(image, boxes, classes, scores, figsize=(7, 7),
                         linewidth=1, color=[0, 0, 1]):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle([x1, y1], w, h,
                              fill=False, edgecolor=color, linewidth=linewidth)
        ax.add_patch(patch)
        ax.text(x1, y1, text, bbox={"facecolor": color, "alpha": 0.4},
                clip_box=ax.clipbox, clip_on=True, )
    plt.show()
    return ax


# TODO: Here is a split line.


def random_flip_horizontal(image, boxes):
    """Flips image and boxes horizontally with 50% chance

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.

    Returns:
      Randomly flipped image and boxes
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes


def resize_and_pad_image(image, min_side=800.0, max_side=1333.0,
                         jitter=[640, 1024], stride=128.0):
    """Resizes and pads image while preserving aspect ratio.

    1. Resizes images so that the shorter side is equal to `min_side`
    2. If the longer side is greater than `max_side`, then resize the image
      with longer side equal to `max_side`
    3. Pad with zeros on right and bottom to make the image shape divisible by
    `stride`

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      min_side: The shorter side of the image is resized to this value, if
        `jitter` is set to None.
      max_side: If the longer side of the image exceeds this value after
        resizing, the image is resized such that the longer side now equals to
        this value.
      jitter: A list of floats containing minimum and maximum size for scale
        jittering. If available, the shorter side of the image will be
        resized to a random value in this range.
      stride: The stride of the smallest feature map in the feature pyramid.
        Can be calculated using `image_size / feature_map_size`.

    Returns:
      image: Resized and padded image.
      image_shape: Shape of the image before padding.
      ratio: The scaling factor used to resize the image
    """
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(
        tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
    )
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )
    return image, image_shape, ratio


def preprocess_data(sample):
    """Applies preprocessing step to a single sample

    Arguments:
      sample: A dict representing a single training sample.

    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """
    image = sample["image"]
    bbox = swap_xy(sample["objects"]["bbox"])
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)

    image, bbox = random_flip_horizontal(image, bbox)
    image, image_shape, _ = resize_and_pad_image(image)

    bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[1],
            bbox[:, 1] * image_shape[0],
            bbox[:, 2] * image_shape[1],
            bbox[:, 3] * image_shape[0],
        ],
        axis=-1,
    )
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id
