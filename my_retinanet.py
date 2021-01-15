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

# from utils.utils import *
# from utils.preprocess import *
from utils.demo_preprocessing import preprocess_data
# from utils.anchorbox import AnchorBox
from utils.demo_labelencoder import LabelEncoder
from utils.demo_resnet50 import get_backbone
from utils.demo_retina_losses import RetinaNetLoss
from utils.demo_retinanet import RetinaNet

# from utils.demo_feature_pyramid import FeaturePyramid
# from utils.network_utils import build_head, get_backbone
# from utils.demo_retina_losses import RetinaNetLoss

# from utils.demo_decode_predictoins import DecodePredictions


"""
## Downloading the COCO2017 dataset

Training on the entire COCO2017 dataset which has around 118k images takes a lot of time, 
hence we will be using a smaller subset of ~500 images for training in this example.

* keras.utils.get_file(): https://keras.io/api/utils/python_utils/#getfile-function
* `RetinaNet-Keras$ python my_retinanet.py`, `os.getcwd()` will return /home/ylqi007/work/PycharmProjects/RetinaNet/RetinaNet-Keras
* `RetinaNet-Keras/utils$ python ../my_retinanet.py`, `os.getcwd()` will return /home/ylqi007/work/PycharmProjects/RetinaNet/RetinaNet-Keras/utils
"""

url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
filename = os.path.join(os.getcwd(), "data.zip")    # this filename is an absolute path.
keras.utils.get_file(filename, url)   # Return path to the downloaded file.

with zipfile.ZipFile("data.zip", "r") as z_fp:
    z_fp.extractall("./")


"""
## Setting up training parameters
"""

model_dir = "retinanet/"
label_encoder = LabelEncoder()

num_classes = 80
batch_size = 2

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)


"""
## Load the COCO2017 dataset using TensorFlow Datasets

- coco/2017: https://www.tensorflow.org/datasets/catalog/coco#coco2017

* Set `data_dir=None` to load the complete dataset
* Load a dataset: https://www.tensorflow.org/datasets/overview#load_a_dataset
* TensorFlow Datasets 数据集载入: https://tf.wiki/zh_hans/appendix/tfds.html#tensorflow-datasets

<_OptionsDataset shapes: {image: (None, None, 3), image/filename: (), image/id: (), objects: {area: (None,), bbox: (None, 4), is_crowd: (None,), label: (None,)}}, types: {image: tf.uint8, image/filename: tf.string, image/id: tf.int64, objects: {area: tf.int64, bbox: tf.float32, is_crowd: tf.bool, label: tf.int64}}>
<_OptionsDataset shapes: {image: (None, None, 3), image/filename: (), image/id: (), objects: {area: (None,), bbox: (None, 4), is_crowd: (None,), label: (None,)}}, types: {image: tf.uint8, image/filename: tf.string, image/id: tf.int64, objects: {area: tf.int64, bbox: tf.float32, is_crowd: tf.bool, label: tf.int64}}>
"""
# Download the complete COCO2017 dataset as tfrecords
# (train_dataset, val_dataset), dataset_info = tfds.load(
#     "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
# )

data_dir = "/home/ylqi007/work/DATA/COCO"
(train_dataset, val_dataset), dataset_info = tfds.load(
    "coco/2017", split=["train", "validation"], with_info=True, data_dir=data_dir)


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

* after labelencoder.encode_batch
    * sample[1], i.e. labels.stack():
 tf.Tensor(
[[[ 449.04822    114.032715    16.029337     7.3659225   -1.       ]
  [ 356.4098      90.50783     14.874092     6.210677    -1.       ]
  [ 282.88266     71.836105    13.718847     5.055431    -1.       ]
  ...
  [-275.89435    -19.222998     7.3649974   -3.0312853   -1.       ]
  [-218.9775     -15.257303     6.209751    -4.1865306   -1.       ]
  [-173.80257    -12.109729     5.0545073   -5.341776    -1.       ]]

 [[ 545.8001     127.00695     16.70464      6.2266526   -1.       ]
  [ 433.2018     100.80548     15.549395     5.0714073   -1.       ]
  [ 343.83252     80.00936     14.39415      3.9161623   -1.       ]
  ...
  [-258.79086    -17.601217     8.0403      -4.1705546   -1.       ]
  [-205.40245    -13.970096     6.885055    -5.3258      -1.       ]
  [-163.02805    -11.088073     5.7298107   -6.4810452   -1.       ]]], shape=(2, 245520, 5), dtype=float32)
"""
autotune = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
train_dataset = train_dataset.shuffle(8 * batch_size)
# padding_values[0] = 0.0, padding value for image
# padding_values[1] = 1e-8, padding value for bboxes
# padding_values[2] = -1, padding value for labels
train_dataset = train_dataset.padded_batch(
    batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
) # images (Batch, H, W, 3); bboxes (Batch, N, 4); class_ids (Batch, N), N=the max num of objects in this batch of images
train_dataset = train_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
train_dataset = train_dataset.prefetch(autotune)

val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
val_dataset = val_dataset.padded_batch(
    batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
val_dataset = val_dataset.prefetch(autotune)


"""
## Initializing and compiling model.
"""
resnet50_backbone = get_backbone()      # resnet50_backbone is a keras.applications.ResNet50
loss_fn = RetinaNetLoss(num_classes)    # The loss function only pass an argument
model = RetinaNet(num_classes=num_classes, backbone=resnet50_backbone)

optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)


"""
## Setting up callbacks.
"""
callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="loss",
        save_best_only=False,
        save_weights_only=True,
        verbose=1
    )
]

# print(callbacks_list)

"""
## Training the model.
"""
epochs = 1

model.fit(
    train_dataset.take(100),
    validation_data=val_dataset.take(50),
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)


# """
# ## Loading weights
# """
#
# # Change this to `model_dir` when not using the downloaded weights
# weights_dir = "data"
# latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
# model.load_weights(latest_checkpoint)
#
#
# """
# ## Building inference model
# """
# image = tf.keras.Input(shape=[None, None, 3], name="image")
# predictions = model(image, training=False)
# detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
# inference_model = tf.keras.Model(inputs=image, outputs=detections)


# """
# ## Generating detections
# """
#
#
# def prepare_image(image):
#     image, _, ratio = resize_and_pad_image(image, jitter=None)
#     image = tf.keras.applications.resnet.preprocess_input(image)
#     return tf.expand_dims(image, axis=0), ratio
#
#
# val_dataset = tfds.load("coco/2017", split="validation", data_dir=data_dir)
# int2str = dataset_info.features["objects"]["label"].int2str
#
#
# for sample in val_dataset.take(2):
#     print(sample)

