#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   Project name: RetinaNet-Keras
#   File name   : networks.py
#   Author      : ylqi007
#   Created date: 2021-01-04 4:28 PM
#
# ================================================================

from tensorflow import keras

from demo_resnet50 import ResNet50

"""
## Building the ResNet50 backbone
RetinaNet uses a ResNet based backbone, using which a feature pyramid network is constructed. In
the example we use ResNet50 as the backbone, and return the feature maps at strides 8, 16 and 32.

keras.application.resnet: https://github.com/keras-team/keras/blob/master/keras/applications/resnet.py
ResNet50: https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py
keras.applications.ResNet50: https://keras.io/api/applications/resnet/#resnet50-function
    * https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py
    * include_top: whether to include the fully-connected layer at the top of the network.
    * input_shape: optional shape tuple, only to be specified if include_top is False (otherwise 
      the input shape has to be (224, 224, 3) or (3, 224, 224)
      
Conv2D layer: https://keras.io/api/layers/convolution_layers/convolution2d/
"""


def get_backbone():
    """Builds ResNet50 with pre-trained imagenet weights"""
    # keras' default ResNet50
    # https://github.com/keras-team/keras/blob/6a46d5259d079a58a9d32ad31a9e9da9c0ea563f/keras/applications/resnet.py#L457
    backbone = keras.applications.ResNet50(include_top=False, input_shape=[224, 224, 3])

    # My utils.demo_resnet50's ResNet50
    # backbone = ResNet50(include_top=False, input_shape=[224, 224, 3])

    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]

    # print(backbone)
    print("c3_output: ", c3_output)
    backbone.summary(line_length=120)

    print('#### backbone.inputs: ', backbone.inputs)

    return keras.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )


get_backbone()
