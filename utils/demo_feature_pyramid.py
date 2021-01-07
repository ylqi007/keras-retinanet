#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   Project name: RetinaNet-Keras
#   File name   : demo_feature_pyramid.py
#   Author      : ylqi007
#   Created date: 2021-01-07 10:37 AM
#
# ================================================================

"""
# Building Feature Pyramid Network as a custom layer.
"""

import tensorflow as tf
from tensorflow import keras
from demo_resnet50 import get_backbone


class FeaturePyramid(keras.layers.Layer):
    """ Builds the Feature Pyramid with the feature maps from the backbone.

    Attributes:
        num_classes: Number of classes in the dataset.
        backbone: The backbone to build the feature pyramid from currently supports ResNet50 only.
    """
    def __init__(self, backbone=None, **kwargs):
        super(FeaturePyramid, self).__init__(name="FeaturePyramid", **kwargs)
        self.backbone = backbone if backbone else get_backbone()
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")   # filters=256, kernel_size=1, strides=1
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, "same")

        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, "same")   # kernel_size=3
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c7_3x3 = keras.layers.Conv2D(256, 3, 1, "same")

        self.upsample_2x = keras.layers.UpSampling2D(2)

    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)  # TODO: training
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)

        # Refer Feature Pyramid paper, ##3.Feature Pyramid Networks
        # Merge bottom-up and top-down maps by element-wise adding
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)

        # Append a 3x3 conv at each merged feature map
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))

        return p3_output, p4_output, p5_output, p6_output, p7_output
