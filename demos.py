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
