#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   Project name: RetinaNet-Keras
#   File name   : demo_net.py
#   Author      : ylqi007
#   Created date: 2021-01-05 9:26 AM
#
# ================================================================

import os
import warnings
from utils.imagenet_utils import _obtain_input_shape
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import models
import tensorflow.keras.utils as keras_utils


WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

# backend = None
# layers = None
# models = None
# keras_utils = None
#
#
# # _KERAS_BACKEND = None
# # _KERAS_LAYERS = None
# # _KERAS_MODELS = None
# # _KERAS_UTILS = None
#
#
# def get_submodules_from_kwargs(kwargs):
#     backend = kwargs.get('backend', _KERAS_BACKEND)
#     layers = kwargs.get('layers', _KERAS_LAYERS)
#     models = kwargs.get('models', _KERAS_MODELS)
#     utils = kwargs.get('utils', _KERAS_UTILS)
#     for key in kwargs.keys():
#         if key not in ['backend', 'layers', 'models', 'utils']:
#             raise TypeError('Invalid keyword argument: %s', key)
#     return backend, layers, models, utils


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """ The identity block is the block that has no conv layer at shortcut. 只接将input加到output

    :param input_tensor: input tensor
    :param kernel_size: default 3, the kernel size of middle conv layer at main path
    :param filters: list of integers, the filters of 3 conv layer at main path
    :param stage: integer, current stage label, used for generating layer names
    :param block: 'a', 'b' ..., current block label, used for generating layer names.

    :return: Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'conv' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Conv2D: 1x1, 64
    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base+'2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    # Conv2D: 3x3, 64
    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    # Conv2D: 3x3, 256
    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # Shortcut
    x = layers.add([x, input_tensor])   # No conv layer at shortcut, i.e. identity shortcut
    x = layers.Activation('relu')(x)

    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """ A block that has a conv layer at shortcut.

    :param input_tensor: input tensor
    :param kernel_size: default 3, the kernel size of middle conv layer at main path
    :param filters: list of integers, the filters of 3 conv layer at main path
    :param stage: integer, current stage label, used for generating layer names
    :param block: 'a','b'..., current block label, used for generating layer names
    :param strides: Strides for the first conv layer in the block.

    :return: Output tensor for the block

    Note that from stage 3, the first conv layer at main path is with strides=(2,2),
    and the shortcut should have strdies=(2,2) as well.
    """
    filters1, filters2, filters3 = filters  # 64, 64, 256 = [64, 64, 256]
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'conv' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Conv2D, 1x1, 64
    x = layers.Conv2D(filters1, (1, 1), strides,        # Conv2D, 1x1, 64
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    # Conv2D, 3x3, 64
    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base+'2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    # Conv2D, 1x1, 64
    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # Shortcut
    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,     # projection shortcut
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis,
                                         name=bn_name_base+'1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ResNet50(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    """ Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is the one specified in Keras config
    at `~/.keras/keras.json`

    :param include_top: whether to include the fully-connected layer at the top of the network.
    :param weights: One of 'None` (random initialization), ==> `None`, i.e. random initialization
        `imagenet` (pre-trained on ImageNet),              ==> `imagenet`, i.e. pre-training
        ot the path to the weights file to be loaded.      ==> path, path to personal weight
    :param input_tensor: optional Keras tensor (i.e. output of the `layers.Input()`) to use as
        image input for the model.
    :param input_shape:
    :param pooling:
    :param classes:
    :param kwargs:
    :return:
    """
    # global backend, layers, models, keras_utils
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either `None` (random initialization), '
                         '`imagenet` (pre-trained on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` as `True`,'
                         '`classes` should be 1000.')   #

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    print('=============== demo_network.input_shape ===============')
    print("input_shape: ", input_shape)

    # Define input layer
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):   # transfer to keras_tensor
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Check image data format
    if backend.image_data_format() == 'channels_last':  # [batch, height, width, channels]
        bn_axis = 3
    else:   # [batch, channels, height, width]
        bn_axis = 1

    print('backend.image_data_format: ', backend.image_data_format())
    print('@@@@ img_input: ', img_input)
    print('#### bn_axis: ', bn_axis)

    # Define backbone
    # Refer paper, Deep Residual Learning for Image Recognition
    # 1st: 7x7, 64, stride=2
    # 2nd: 3x3 max pool, stride=2
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),       # num of params = 7*7*3*64+64=9472
                      strides=(2, 2),
                      padding='valid',  # `valid` means no padding
                      kernel_initializer='he_normal',
                      name='conv1')(x)  # output_shape = floor(i+2p-k)/s + 1 = (230+0-7)/2+1=112
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)  # output=floor(i-k)s+1=(114-3)/2+1=56

    # residual block, stage 2, 9 convs
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1)) # O=(B, 56, 56, 256)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')     # O=(B, 56, 56, 256)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')     # O=(B, 56, 56, 256)

    # residual block, stage 3, 12 convs
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a') # default strides=(2,2), O=(None, 28, 28, 512)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')   # O=(None, 28, 28, 512)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')   # O=(None, 28, 28, 512)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')   # O=(None, 28, 28, 512)

    # redidual block, stage 4, 6x3=18 convs
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')    # strides=(2,2), O=(None, 14, 14, 1024)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')  # O=(None, 14, 14, 1024)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')  # O=(None, 14, 14, 1024)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')  # O=(None, 14, 14, 1024)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')  # O=(None, 14, 14, 1024)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')  # O=(None, 14, 14, 1024)

    # residual block, stage 5, 9 convs
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')  # strides=(2,2), O=(None, 7, 7, 2048)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')  # O=(None, 7, 7, 2048)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')  # O=(None, 7, 7, 2048)

    # Define top heads
    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)   # classes=1000
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalAveragePooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` has been '
                          'changed since Keras 2.2.0.')

    # Ensure that the model takes into account any potential predecessors of `input_tensor`
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model
    model = models.Model(inputs, x, name='resnet01')

    # Load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model


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
    backbone = keras.applications.ResNet50(include_top=False, input_shape=[None, None, 3])

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
