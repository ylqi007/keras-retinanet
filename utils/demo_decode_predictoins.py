#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   Project name: RetinaNet-Keras
#   File name   : demo_decode_predictoins.py
#   Author      : ylqi007
#   Created date: 2021-01-07 4:32 PM
#
# ================================================================

import tensorflow as tf
from utils.utils import convert_to_corners
from utils.anchorbox import AnchorBox

"""
# Implementing a custom layer to decode predictions.
"""

class DecodePredictions(tf.keras.layers.Layer):
    """ A Keras layer that decodes predictions of the RetinaNet model.

    """

    def __init__(self,
                 num_classes=80,
                 confidence_threshold=0.05,
                 nms_iou_threshold=0.5,
                 max_detections_per_class=100,
                 max_detections=100,
                 box_variance=[0.1, 0.1, 0.2, 0.2],
                 **kwargs):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor([0.1, 0.1, 0.2, 0.3],
                                                  dtype=tf.float32)

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:]
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed
        pass

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2]) # height, width
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])  # sigmoid
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False
        )
