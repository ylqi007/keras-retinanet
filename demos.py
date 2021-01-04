#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   Project name: RetinaNet-Keras
#   File name   : demos.py
#   Author      : ylqi007
#   Created date: 2021-01-01 11:13 PM
#
# ================================================================

import numpy as np
import tensorflow as tf

"""
# check stack()

boxes.shape=(3, 4)
boxes[:, 0].shape=(3,)
boxes[0, :].shape=(4,)
"""
boxes = np.array([[0.5101, 0.2713, 0.8426, 0.8679],
                  [0.1529, 0.5445, 0.5799, 0.7872],
                  [0.0067, 0.2022, 0.1865, 0.3775]])

boxes_column0 = boxes[:, 0]
boxes_column1 = boxes[:, 1]
boxes_column2 = boxes[:, 2]
boxes_column3 = boxes[:, 3]
print(type(boxes_column3))
print(np.array([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]]).shape)

stack1 = np.stack([boxes_column1, boxes_column0])
stack2 = np.stack([boxes_column1, boxes_column0], axis=-1)
print(stack1, stack1.shape)
print(stack2, stack2.shape)


"""
# Check concat()

* [xmin, ymin, xmax, ymax] to [x, y, width, height]
* axis=0, concate (None, 2) with (None, 2) to (None, 2)
* axis=-1, concate (None, 2) with (Nonde, 2) to (None, 4)
"""

converted_bbox = np.concatenate(
    [(boxes[..., :2] + boxes[..., 2:]) / 2.0, (boxes[..., 2:] - boxes[..., :2]) / 2.0],
    axis=-1)
print(converted_bbox)


# TODO: Splitline -- AnchorBox._compute_dims()
def _compute_dims():
    """
    anchor_dims:
         [<tf.Tensor: shape=(1, 1, 2), dtype=float32, numpy=array([[[ 8., 16.]]], dtype=float32)>, ==> area=8, ratio=0.5, scale=4
         <tf.Tensor: shape=(1, 1, 2), dtype=float32, numpy=array([[[16., 32.]]], dtype=float32)>, ==> area=8, ratio=0.5, scale=8
         <tf.Tensor: shape=(1, 1, 2), dtype=float32, numpy=array([[[11.313708, 11.313708]]], dtype=float32)>, ==> area=8, ratio=1, scale=4
         <tf.Tensor: shape=(1, 1, 2), dtype=float32, numpy=array([[[22.627417, 22.627417]]], dtype=float32)>]  ==> area=8, ratio=1, scale=4
        * Above 4 tf.Tensor is for area=8

    tf.stack(anchor_dims, axis=-2):
         tf.Tensor(
            [[[[ 8.       16.      ]
               [16.       32.      ]
               [11.313708 11.313708]
               [22.627417 22.627417]]]], shape=(1, 1, 4, 2), dtype=float32)
        * stack 4 tensors into 1 tensor, with axis=-2 equals number of tensors

    anchor_dims_all:
         [<tf.Tensor: shape=(1, 1, 4, 2), dtype=float32, numpy=     ==> area=8
            array([[[[ 8.      , 16.      ],                            ==> area=8, ratio=0.5
                     [16.      , 32.      ],
                     [11.313708, 11.313708],                            ==> area=8, ratio=1
                     [22.627417, 22.627417]]]], dtype=float32)>,
        <tf.Tensor: shape=(1, 1, 4, 2), dtype=float32, numpy=       ==> area=16
            array([[[[11.313708, 22.627417],                            ==> area=16, ratio=0.5
                     [22.627417, 45.254833],
                     [16.      , 16.      ],                            ==> area=16, ratio=1
                     [32.      , 32.      ]]]], dtype=float32)>,
        <tf.Tensor: shape=(1, 1, 4, 2), dtype=float32, numpy=       ==> area=32
            array([[[[16.      , 32.      ],                            ==> area=32, ratio=0.5
                     [32.      , 64.      ],
                     [22.627417, 22.627417],                            ==> area=32, ratio=01
                     [45.254833, 45.254833]]]], dtype=float32)>]
    :return:
    """
    anchor_dims_all = []
    for area in [8, 16, 32]:
        anchor_dims = []    # all anchor_dims for a specific area
        for ratio in [0.5, 1.0]:
            anchor_height = tf.math.sqrt(area / ratio)
            anchor_width = area / anchor_height
            print('[anchor_width, anchor_height]:\n', [anchor_width, anchor_height])
            print('stack([anchor_width, anchor_height]):\n', tf.stack([anchor_width,
                                                                       anchor_height], axis=-1))
            dims = tf.reshape(
                tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
            )
            print('dims:\n', dims)
            for scale in [4, 8]:
                anchor_dims.append(dims * scale)
            print('anchor_dims:\n', anchor_dims)
        print('@@@@\n', anchor_dims)
        print('####\n', tf.stack(anchor_dims, axis=-2))
        anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        print('$$$$\n', anchor_dims_all)
    return anchor_dims_all


print("\n==========================================================")
print("========== _compute_dims() ==========")
_compute_dims()


# =============================================================================
# _get_anchor()
# =============================================================================
"""
# _get_anchor()
# Check meshgrid

* rx: tf.Tensor([0.5 1.5 2.5], shape=(3,), dtype=float32)
* ry: tf.Tensor([0.5 1.5 2.5 3.5], shape=(4,), dtype=float32)

* centers after meshgridding:
    [<tf.Tensor: shape=(4, 3), dtype=float32, numpy=
    array([[0.5, 1.5, 2.5],
           [0.5, 1.5, 2.5],
           [0.5, 1.5, 2.5],
           [0.5, 1.5, 2.5]], dtype=float32)>, 
     <tf.Tensor: shape=(4, 3), dtype=float32, numpy=
    array([[0.5, 0.5, 0.5],
           [1.5, 1.5, 1.5],
           [2.5, 2.5, 2.5],
           [3.5, 3.5, 3.5]], dtype=float32)>]
* shape of the output of meshgrid: (4, 3)

* centers after stacking 
tf.Tensor(
[[[0.5 0.5]
  [1.5 0.5]
  [2.5 0.5]]

 [[0.5 1.5]
  [1.5 1.5]
  [2.5 1.5]]

 [[0.5 2.5]
  [1.5 2.5]
  [2.5 2.5]]

 [[0.5 3.5]
  [1.5 3.5]
  [2.5 3.5]]], shape=(4, 3, 2), dtype=float32)

center_expand_dims:
 tf.Tensor(
[[[[0.5 0.5]]
  [[1.5 0.5]]
  [[2.5 0.5]]]

 [[[0.5 1.5]]
  [[1.5 1.5]]
  [[2.5 1.5]]]

 [[[0.5 2.5]]
  [[1.5 2.5]]
  [[2.5 2.5]]]

 [[[0.5 3.5]]
  [[1.5 3.5]]
  [[2.5 3.5]]]], shape=(4, 3, 1, 2), dtype=float32)

center_tiled:
 tf.Tensor(
[[[[0.5 0.5]
   [0.5 0.5]
   [0.5 0.5]
   [0.5 0.5]
   [0.5 0.5]]

  [[1.5 0.5]
   [1.5 0.5]
   [1.5 0.5]
   [1.5 0.5]
   [1.5 0.5]]

  [[2.5 0.5]
   [2.5 0.5]
   [2.5 0.5]
   [2.5 0.5]
   [2.5 0.5]]]


 [[[0.5 1.5]
   [0.5 1.5]
   [0.5 1.5]
   [0.5 1.5]
   [0.5 1.5]]

  [[1.5 1.5]
   [1.5 1.5]
   [1.5 1.5]
   [1.5 1.5]
   [1.5 1.5]]

  [[2.5 1.5]
   [2.5 1.5]
   [2.5 1.5]
   [2.5 1.5]
   [2.5 1.5]]]


 [[[0.5 2.5]
   [0.5 2.5]
   [0.5 2.5]
   [0.5 2.5]
   [0.5 2.5]]

  [[1.5 2.5]
   [1.5 2.5]
   [1.5 2.5]
   [1.5 2.5]
   [1.5 2.5]]

  [[2.5 2.5]
   [2.5 2.5]
   [2.5 2.5]
   [2.5 2.5]
   [2.5 2.5]]]


 [[[0.5 3.5]
   [0.5 3.5]
   [0.5 3.5]
   [0.5 3.5]
   [0.5 3.5]]

  [[1.5 3.5]
   [1.5 3.5]
   [1.5 3.5]
   [1.5 3.5]
   [1.5 3.5]]

  [[2.5 3.5]
   [2.5 3.5]
   [2.5 3.5]
   [2.5 3.5]
   [2.5 3.5]]]], shape=(4, 3, 5, 2), dtype=float32)

"""


def _get_anchors(feature_height=4, feature_width=3, level=3):
    rx = tf.range(feature_width, dtype=tf.float32) + 0.5
    ry = tf.range(feature_height, dtype=tf.float32) + 0.5
    print('rx: \n', rx)
    print('ry: \n', ry)
    print('\n')
    centers = tf.meshgrid(rx, ry)
    print('centers after meshgridding:\n', centers)     # shape = [shape=(4,3), shape=(4,3)]
    print('\n')
    centers = tf.stack(centers, axis=-1)        # after stacking: shape = (4, 3, 2)
    print('centers after stacking\n', centers)
    center_expand_dims = tf.expand_dims(centers, axis=-2)   # shape = (4, 3, 1, 2)
    print('center_expand_dims:\n', center_expand_dims)
    _num_anchors = 4
    center_tiled = tf.tile(center_expand_dims, [1, 1, _num_anchors, 1]) # shape=(4, 3, 5, 2)
    print('center_tiled:\n', center_tiled)

    # dims
    dims = tf.tile(_compute_dims()[0], [feature_height, feature_width, 1, 1])
    print('#### dims:\n', dims.shape)
    print('#### cens:\n', center_tiled.shape)

    # anchors
    anchors = tf.concat([center_tiled, dims], axis=-1)  # shape=(fea_height, fea_width, #_anchors, 4]
    print('@@@@ anchors:\n', anchors)

    result = tf.reshape(
        anchors, [feature_height * feature_width * _num_anchors, 4]
    )
    print('$$$$ result:\n', result)
    return tf.reshape(
        anchors, [feature_height * feature_width * _num_anchors, 4]
    )


print('=============================================================================')
_get_anchors()


def get_anchors(image_height=5, image_width=5):
    anchors = [_get_anchors(tf.math.ceil(image_height / 2 ** i),
                            tf.math.ceil(image_width / 2 ** i),
                            i) for i in range(3, 6)]
    print('#### anchors:\n', anchors)
    result = tf.concat(anchors, axis=0)
    print('@@@@ concate:\n', result)


print('=============================================================================')
get_anchors()


"""
# Compute IoU
"""
anchor = np.array([[0.5101, 0.2713, 0.8426, 0.8679],
                  [0.1529, 0.5445, 0.5799, 0.7872],
                  [0.0067, 0.2022, 0.1865, 0.3775]])
gt_boxes = np.reshape(np.arange(20), (-1, 4))


def compute_iou(boxes1=anchor, boxes2=gt_boxes):
    """
    Compute pairwise IoU matrix for given tow sets of boxes.

    :param boxes1: A tensor with shape `(M, 4)` representing bounding boxes where each box is of
        the format `[x, y, width, height]`.
    :param boxes2: A tensor with shape `(N, 4)` representing bounding boxes where each box is of
        the format `[x, y, width, height]`.

    :return: pairwise IoU matrix with shape `(M, N)`, where the values at ith row jth column
        holds the IoU between ith box and jth box from boxes1 and boxes2 respectively.
    """
    # boxes1_corners = convert_to_corners(boxes1) # [x, y, width, height] ==> [xmin, ymin, xmax, ymax]
    # boxes2_corners = convert_to_corners(boxes2) # [x, y, width, height] ==> [xmin, ymin, xmax, ymax]
    print('boxes1[:, None, :2]\n', boxes1[:, None, :2].shape)   # boxes1: (M, 4)
    print('boxes2[:, :2]\n', boxes2[:, :2].shape)               # boxes2: (N, 4)
    lu = tf.maximum(boxes1[:, None, :2], boxes2[:, :2])         # lu: (M, N, 2), [[[max(xmin), max(ymin)]...]]
    rd = tf.minimum(boxes1[:, None, 2:], boxes2[:, 2:])         # rd: (M, N, 2)
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]   # [M, N, w] * [M, N,
    # h] => (M, N)
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]   # boxes1 is of format [x, y, w, h], area.shape=(M,)
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]   # boxes2_area is of format (N,)
    print('shape of intersection_area:\t', intersection_area.shape)
    print('shape of boxes2_area:\t\t', boxes2_area.shape)
    print('shape of boxes1_area[:, None]:\t', boxes1_area[:, None].shape)
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    print(union_area)
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)


print('=============================================================================')
compute_iou()