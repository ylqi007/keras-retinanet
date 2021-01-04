#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   Project name: RetinaNet-Keras
#   File name   : labelencoder.py
#   Author      : ylqi007
#   Created date: 2021-01-03 8:35 AM
#
# ================================================================

import tensorflow as tf

from utils.utils import compute_iou
from utils.anchorbox import AnchorBox


"""
## Encoding labels

The raw labels, consisting of bounding boxes and class ids need to be transformed into targets
for training. This transformation consists of the following steps:
- Generating anchor boxes for the given image dimensions.
- Assigning ground truth boxes to the anchor boxes.
- The anchor boxes that are not assigned any objects, are either assigned the background class or 
ignored depending on the IoU.
- Generating the class classification and regression targets using anchor boxes.
"""


class LabelEncoder:
    """
    Transform the raw labels into targets for training.
    This class has operations to generate targets for a batch of samples which is made up of the
    input image, bounding boxes for the objects present and their class ids.

    Attributes:
        anchor_box: Anchor box generate to encode the bounding boxes.
        box_variance: The scaling factors used to scale the bounding box targets.
    """
    def __init__(self):
        self._anchor_box = AnchorBox()  # anchor box generator
        self._box_variance = tf.convert_to_tensor([0.1, 0.1, 0.2, 0.2], dtype=tf.float32)

    def _match_anchor_boxes(self, anchor_boxes, gt_boxes,
                            match_iou=0.5, ignore_iou=0.4):
        """
        Matches ground truth boxes to anchor boxes based on IoU.
        1. Calculates the pairwise IoU for the M `anchor_boxes` and N target `gt_boxes` to get a
            (M, N) shaped matrix.
        2. The ground truth box with the maximum IoU in each row is assigned to the anchor box
            provided the IoU is greater than `match_iou`.
        3. If the maximum IoU in a row is less than `ignore_iou`, the anchor box is assigned with
            the background class.
        4. The remaining anchor boxes that do not have any class assigned are ignored during
            training.

        :param anchor_boxes: A float tensor with the shape `(total_anchors, 4)` representing all
            the anchor boxes for a given input image shape, where each anchor box is of the
            format [x, y, width height]. total_anchors = M
        :param gt_boxes: A float tensor with shape (num_objects, 4) representing the ground truth
            boxes, where each box is of the format [x, y, width, height]. num_objects = N
        :param match_iou: A float value representing the minimum IoU threshold for determining
            if a ground truth can be assigned to an anchor box.
        :param ignore_iou: A float value representing the IoU threshold under which an anchor box
            is assigned to the background class.

        :return:

        Match anchor boxes to gt_boxes, many anchor boxes may mapped to one gt_box
        """
        # TODO: matched_gt_idx
        print("================== LabelEncoder._match_anchor_boxes() ================")
        iou_matrix = compute_iou(anchor_boxes, gt_boxes)    # (M, N), M anchor_boxes, N gt_boxes
        # match each anchor box to a gt_box
        max_iou = tf.reduce_max(iou_matrix, axis=1)         # (M,), keep M anchor boxes' IoU
        # the index of matched gt_box
        # matched_git_idx: Tensor("while/ArgMax:0", shape=(None,), dtype=int64)
        # positive_mask:   Tensor("while/GreaterEqual:0", shape=(None,), dtype=bool)
        # negative_mask:   Tensor("while/Less:0", shape=(None,), dtype=bool)
        # ignore_mask:     Tensor("while/LogicalNot:0", shape=(None,), dtype=bool)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)      # (M,), get idx of max IoU in each row
        positive_mask = tf.greater_equal(max_iou, match_iou)    # the max_iou >= match_iou
        negative_mask = tf.less(max_iou, ignore_iou)        # the max_iou < ignore_iou
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return (matched_gt_idx,     # idx means this anchor match which gt_boxes
                tf.cast(positive_mask, dtype=tf.float32),
                tf.cast(ignore_mask, dtype=tf.float32))

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """ Transforms the ground truth boxes into targets for training.

        :param anchor_boxes: [xmin, ymin, xmax, ymax] ?, [x, y, w, h] ? seems [x, y, w, h]
        :param matched_gt_boxes:
        :return:
        """
        # TODO: try to understand
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1
        )
        box_target = box_target / self._box_variance    # _box_variance = [0.1, 0.1, 0.2, 0.2]
        return box_target

    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        """ Creates box and classification targets for a single sample.

        :param image_shape: [Batch, height, width, channel=3]; #TODO: or [:, width, height, :]
        :param gt_boxes: [Batch, N, 4], where N is the max number of objects in this batch.
        :param cls_ids: [Batch, N]
        :return:

        * self._anchor_box = AnchorBox()
        * image_shape[1]: height; image_shape[2]: width
        """
        print("================== LabelEncoder._encode_sample() ================")
        # anchor_boxes.shape = ((num_anchors0 + num_anchors1 + num_anchors2 + ...), 4)
        # num_anchor0 = tf.math.ceil(image_height / 2 ** 3) * len(area) * len(ratio)
        # M = num_anchor0 + num_anchor1 + num_anchor2 + ...
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])  # shape=(M,4)
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)    # cls_ids.shape = (None,)
        # anchor_boxes.shape:    (None, 4)
        # cls_ids.shape:         (None,)
        # matched_gt_idx.shape:  (None,)
        # positive_mask.shape:   (None,)
        # ignore_mask.shape:     (None,)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(anchor_boxes,
                                                                              gt_boxes)
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)  # matched_gt_idx.shape = (M,)
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)   # gt_box to anchor
        # cls_ids:              Tensor("while/Cast:0", shape=(None,), dtype=float32)
        # matched_gt_cls_ids:   Tensor("while/GatherV2_1:0", shape=(None,), dtype=float32)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(
            tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids
        )   # Tensor("while/SelectV2:0", shape=(None,), dtype=float32)
        # cls_target after th.where(): Tensor("while/SelectV2_1:0", shape=(None,), dtype=float32)
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)  # -2.0, ignored anchor
        cls_target = tf.expand_dims(cls_target, axis=-1)
        # box_target: Tensor("while/truediv_17:0", shape=(None, 4), dtype=float32)
        # cls_target: Tensor("while/ExpandDims_5:0", shape=(None, 1), dtype=float32)
        label = tf.concat([box_target, cls_target], axis=-1)
        return label

    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        """ Creates box and classification targets for a batch.

        :param batch_images:
        :param gt_boxes:
        :param cls_ids:
        :return:

        tf.shape(): https://www.tensorflow.org/api_docs/python/tf/shape
            * tf.shape(batch_images) ==> Tensor("Shape:0", shape=(4,), dtype=int32)
            * batch_images.shape ==> (2, None, None, 3), where 2 = batch_size
        tf.TensorArray(): https://www.tensorflow.org/api_docs/python/tf/TensorArray
        """
        print("================== LabelEncoder.encode_batch() ================")
        images_shape = tf.shape(batch_images)   # Tensor("Shape:0", shape=(4,), dtype=int32)
        batch_size = images_shape[0]    # Tensor("strided_slice:0", shape=(), dtype=int32)
        # <tensorflow.python.ops.tensor_array_ops.TensorArray object at 0x7fd55435f520>
        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            # images_shape:         Tensor("Shape:0", shape=(4,), dtype=int32)
            # images_shape.shape:   (4,)
            # gt_boxes[i].shape:    (None, 4), N is the max number of objects in this batch, it's not a fixed value here.
            # cls_ids[i].shape:     (None,), N is the max number of objects in this batch
            # print('@#$: images_shape:\t', images_shape, images_shape.shape)
            # print('@#$: gt_boxes[i]:\t', gt_boxes[i].shape)
            # print('@#$: cls_ids[i]:\t', cls_ids[i].shape)
            label = self._encode_sample(images_shape, gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
        batch_images = tf.keras.applications.resnet.preprocess_input(batch_images)
        # labels:  <tensorflow.python.ops.tensor_array_ops.TensorArray object at 0x7f175c29c2e0>
        # labels.stack():  Tensor("TensorArrayV2Stack/TensorListStack:0", shape=(None, None, 5), dtype=float32)
        return batch_images, labels.stack()
