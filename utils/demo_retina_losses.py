#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   Project name: RetinaNet-Keras
#   File name   : demo_retina_losses.py
#   Author      : ylqi007
#   Created date: 2021-01-09 2:23 PM
#
# ================================================================

import tensorflow as tf

"""
# Implementing Smooth L1 loss and Focal loss as keras custom losses.

tf.keras.losses.Loss: https://github.com/tensorflow/tensorflow/blob/582c8d236cb079023657287c318ff26adb239002/tensorflow/python/keras/losses.py#L47-L210

"""


class RetinaNetBoxLoss(tf.losses.Loss):
    """ Implements Smooth L1 loss.

    Refer: Fast R-CNN
    """
    def __init__(self, delta):
        super(RetinaNetBoxLoss, self).__init__(reduction='none',
                                               name='RetinaNetBoxLoss')
        self._delta = delta

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference ** 2
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetClassificationLoss(tf.losses.Loss):
    """ Implements Focal loss.

    Refer: Focal Loss for Dense object Detection.
    """
    def __init__(self, alpha, gamma):
        super(RetinaNetClassificationLoss, self).__init__(
            reduction="none", name="RetinaNetClassificationLoss"
        )
        self._alpha = alpha
        self._gamma = gamma

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                                logits=y_pred)
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, (1.0 - probs))
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetLoss(tf.losses.Loss):
    """ Wrapper to combine both the classification and box regression losses.

    Box regression loss: Smooth L1 loss.
    Classification loss: Focal Loss.
    """
    def __init__(self, num_classes=80, alpha=0.25, gamma=2.0, delta=1.0):
        super(RetinaNetLoss, self).__init__(reduction="auto", name="RetinaNetLoss")
        self._clf_loss = RetinaNetClassificationLoss(alpha, gamma)
        self._box_loss = RetinaNetBoxLoss(delta)
        self._num_classes = num_classes

    def call(self, y_true, y_pred):
        """

        :param y_true: Tensor("IteratorGetNext:1", shape=(None, None, 5), dtype=float32)
        :param y_pred: Tensor("RetinaNet/concat_2:0", shape=(2, 10143, 84), dtype=float32)
            * since the input shape is [224, 224, 3], therefore, it is 10143 here.
        :return:
        """
        print("============ RetinaLoss.call() ===============")
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        # ground truth boxes and predicted boxes
        box_labels = y_true[:, :, :4]       # i.e. truth boxes coordinates, Tensor("RetinaNetLoss/strided_slice:0", shape=(None, None, 4), dtype=float32)
        box_predictions = y_pred[:, :, :4]  # i.e. predicted boxes coordinates, Tensor("RetinaNetLoss/strided_slice_1:0", shape=(2, 10143, 4), dtype=float32)

        # classes, (None, None, 1) ==> (None, None, num_classes)
        cls_labels = tf.one_hot(tf.cast(y_true[:, :, 4], dtype=tf.int32),   # Tensor("RetinaNetLoss/one_hot:0", shape=(None, None, 80), dtype=float32)
                                depth=self._num_classes,
                                dtype=tf.float32)   # one-hot encode cls integer representation
        cls_predictions = y_pred[:, :, 4:]      # Tensor("RetinaNetLoss/strided_slice_3:0", shape=(2, 10143, 80), dtype=float32)

        # masks, >=0 is positive case, -1 is negative case, and -2 is ignored case
        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)    # Tensor("RetinaNetLoss/Cast_1:0", shape=(None, None), dtype=float32)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)        # -2, ignore cases, Tensor("RetinaNetLoss/Cast_2:0", shape=(None, None), dtype=float32)

        # losses
        print("=============== demo_retina_losses ")
        clf_loss = self._clf_loss(cls_labels, cls_predictions)  # Tensor("RetinaNetLoss/RetinaNetClassificationLoss/weighted_loss/Mul:0", shape=(2, 10143), dtype=float32)
        box_loss = self._box_loss(box_labels, box_predictions)  # Tensor("RetinaNetLoss/RetinaNetBoxLoss/weighted_loss/Mul:0", shape=(2, 10143), dtype=float32)
        # print("#### clf_loss: ", clf_loss)
        # print("#### box_loss: ", box_loss)
        # print("$$$$ ignore_mask: ", ignore_mask)
        # print("$$$$ positive_mask: ", positive_mask)
        # clf_loss: positive loss + negative loss
        # box_loss: only positive loss
        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)      # set ignored loss to 0
        print("## positive_mask: ", positive_mask)
        print("## box_loss: ", box_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)    # set unmatched loss to 0

        # normalized loss
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        loss = clf_loss + box_loss

        return loss
