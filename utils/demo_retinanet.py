#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   Project name: RetinaNet-Keras
#   File name   : demo_retinanet.py
#   Author      : ylqi007
#   Created date: 2021-01-09 11:10 PM
#
# ================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.demo_feature_pyramid import FeaturePyramid
from utils.network_utils import build_head

"""
## Building RetinaNet using a subclassed model.
"""


class RetinaNet(keras.Model):
    """ A subclassed keras model implementing the RetinaNet architecture.

    Attributes:
        * num_classes: Number of classes in the dataset.
        * backbone: The backbone to build the feature pyramid from current supports ResNet50 only.
    """
    def __init__(self, num_classes, backbone=None, **kwargs):
        super(RetinaNet, self).__init__(name='RetinaNet', **kwargs)
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")  # A=9, i.e. 9 anchors

    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        print("========= utils.demo_retinanet.RetinaNet.call() ==========")
        print(features)
        N = tf.shape(image)[0]      # batch size
        cls_outputs = []
        box_outputs = []

        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))  # # of total anchors
            cls_outputs.append(tf.reshape(self.cls_head(feature), [N, -1, self.num_classes]))

        cls_outputs = tf.concat(cls_outputs, axis=1)    # Tensor("RetinaNet/concat:0", shape=(2, 10143, 80), dtype=float32)
        box_outputs = tf.concat(box_outputs, axis=1)    # Tensor("RetinaNet/concat_1:0", shape=(2, 10143, 4), dtype=float32)

        return tf.concat([box_outputs, cls_outputs], axis=-1)   # Tensor("RetinaNet/concat_2:0", shape=(2, 10143, 84), dtype=float32)
