#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   Project name: RetinaNet-Keras
#   File name   : anchor.py
#   Author      : ylqi007
#   Created date: 2021-01-01 1:28 PM
#
# ================================================================
import os
import re
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


"""
## Implementing Anchor boxes generator

Anchor boxes are fixed sized boxes that the model uses to predict the bounding box for an object. 
It does this by regressing the offset between the locationi of the object's center and the center 
of an anchor box, and then uses the width and height of the anchor box to predict a relative 
scale of the object. In the case of RetinaNet, each location on a given feature map has nine 
anchor boxes (at three scales and three ratios).
"""


class AnchorBox:
    """ Generates anchor boxes.

    This class has opeations to generate anchor boxes for feature maps at strides `[8, 16, 32,
    64, 128]`. Where each anchor each box is of the format `[x, y, width, height]`.

    Attributes:
        aspect_ratios: A list of float values representing the aspect ratios of the anchor boxes
            at each location on the feature map.
        scales: A list of float values representing the scale of the anchor boxes at each
            location on the feature map.
        num_anchors: The number of anchor boxes at each location on feature map.
        areas: A list of float value representing the areas of the anchor boxes for each feature
            map in the feature pyramid.
        strides: A list of float value representing the strides for each feature map in the
            featreu pyramid.
    """
    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1/3, 2/3]]       # [2^0, 2^(1/3), 2^(2/3)]
        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]       # [2^3, ..., 2^7]
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]   # [32^2, ..., 512^2]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """
        Computes anchor box dimensions for all ratios and scales at all levels of the feature
        pyramid.
        :return:
        """
        anchor_dims_all = []
        for area in self._areas:    # _areas = [32^2, 64^2, ..., 512^2]
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = tf.math.sqrt(area / anchor_height)
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )   # dim0=1, i.e area; dim1=1, i.e ratio; dim2=2, i.e. [width, height]
                for scale in self.scales:
                    anchor_dims.append(scale * dims)    # [shape=(1,1,2), shape=(1,1,2), ...]
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        # anchor_dims_all is a list of tensors with shape=(1, 1, len(ratios)*len(scales), 2), ...]
        return anchor_dims_all      # [shape=(1, 1, len(ratios)*len(scales), 2), ...]

    def _get_anchors(self, feature_height, feature_width, level):
        """
        Generate anchor boxes for a given feature map size and level. (计算一个feature map的anchors)

        centers after tf.tile(...): shape=(len(ry), len(rx), num_anchors, 2)
        _anchor_dims: [(1, 1, # of anchors, 2), (1, 1, num_anchors, 2), ...]
        _anchor_dims[level-3]: shape=(1, 1, # of anchors, 2)
        _anchor_dimes[level-3] after tiling: shape=(feature_h, feature_w, num_anchors, 2)
        anchors = tf.concat([centers, dims], axis=-1): shape=(feature_h, feature_w, num_anchors, 4]

        :param feature_height: An integer representing the height of the feature map.
        :param feature_width: An integer representing the width of the feature map.
        :param level: An integer representing the level of the feature map in feature pyramid.
        :return: anchor boxes with the shape `(feature_height * feature_width * num_anchors, 4)`
        """
        print("================== AnchorBox._get_anchors() ================")
        # indexes of x-axis and y-axis
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5    # Tensor("while/add_2:0", shape=(None,), dtype=float32), i.e. a vector
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        # centers of each cell after striding, shape=(len(ry), len(rx), 2)
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)      # shape=(len(ry), len(rx), 1, 2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1]) # shape=(len(ry), len(rx), num_anchors, 2)
        # dims in each cell
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )   # shape=(feature_h, feature_w, # of anchors, 2)
        # concat centers and dims, then get anchors
        anchors = tf.concat([centers, dims], axis=-1)   # shape=(feature_h, feature_w, num_anchors, 4]
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )   # shape=(feature_h * feature_w * * num_anchors, 4)

    def get_anchors(self, image_height, image_width):
        """
        Generates anchor boxes for all the feature maps of the feature pyramid.

        :param image_height: Height of the input image.
        :param image_width: Width of the input image.
        :return: anchor boxes for all the feature maps, stacked as a single tensor with shape
        `(toral_anchors, 4)`
        """
        # print("================== AnchorBox.get_anchors() ================")
        # tf.math.ceil(image_height / 2 ** i), the height of i-th feature map
        # tf.math.ceil(image_width / 2 ** i), the width of i-th feature map
        # i, the level of the feature map
        # image_height:    Tensor("while/strided_slice_2:0", shape=(), dtype=int32)
        # image_width:     Tensor("while/strided_slice_3:0", shape=(), dtype=int32)
        # shape=(), i.e. a scalar
        anchors = [self._get_anchors(tf.math.ceil(image_height / 2 ** i),
                                     tf.math.ceil(image_width / 2 ** i),
                                     i) for i in range(3, 8)]
        # after concating: shape = ((num_anchors0 + num_anchors1 + num_anchors2 + ...), 4)
        return tf.concat(anchors, axis=0)
