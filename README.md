[toc]

# RetinaNet-Keras

## Load tf.data.Dataset 

* [tfds -- COCO](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/object_detection/coco.py)
* The original dataset after `tfds.load(...)`


```python
@@@@ sample["image"]:
 <class 'tensorflow.python.framework.ops.EagerTensor'> (462, 640, 3)

@@@@ sample["image/filename"]:
 <class 'tensorflow.python.framework.ops.EagerTensor'> ()

@@@@ sample["image/id"]:
 <class 'tensorflow.python.framework.ops.EagerTensor'> ()

@@@@ sample["objects"]:
 <class 'dict'> {'area': <tf.Tensor: shape=(3,), dtype=int64, numpy=array([17821, 16942,  4344])>, 'bbox': <tf.Tensor: shape=(3, 4), dtype=float32, numpy=
array([[0.54380953, 0.13464062, 0.98651516, 0.33742186],

​       [0.50707793, 0.517875  , 0.8044805 , 0.891125  ],

​       [0.3264935 , 0.36971876, 0.65203464, 0.4431875 ]], dtype=float32)>, 'is_crowd': <tf.Tensor: shape=(3,), dtype=bool, numpy=array([False, False, False])>, 'label': <tf.Tensor: shape=(3,), dtype=int64, numpy=array([3, 3, 0])>}

@@@@ sample["objects"]["area"]:
 <class 'tensorflow.python.framework.ops.EagerTensor'> tf.Tensor([17821 16942  4344], shape=(3,), dtype=int64)

@@@@ sample["objects"]["bbox"]:
 <class 'tensorflow.python.framework.ops.EagerTensor'> tf.Tensor(
[[0.54380953 0.13464062 0.98651516 0.33742186]
 [0.50707793 0.517875   0.8044805  0.891125  ]
 [0.3264935  0.36971876 0.65203464 0.4431875 ]], shape=(3, 4), dtype=float32)

@@@@ sample["objects"]["is_crowd"]:
 <class 'tensorflow.python.framework.ops.EagerTensor'> tf.Tensor([False False False], shape=(3,), dtype=bool)
@@@@ sample["objects"]["label"]:
 <class 'tensorflow.python.framework.ops.EagerTensor'> tf.Tensor([3 3 0], shape=(3,), dtype=int64)
```


## train_dataset.padded_batch

```python
sample[0], i.e. images:
 <class 'tensorflow.python.framework.ops.EagerTensor'> (2, 1024, 1280, 3)
sample[1], i.e. bboxes:
 <class 'tensorflow.python.framework.ops.EagerTensor'> (2, 46, 4)
sample[2], i.e. class_ids:
 <class 'tensorflow.python.framework.ops.EagerTensor'> (2, 46)
```

## train_dataset.map(LabelEncoder.encode_batch, ...)

```python
sample[0], i.e. batch_images:
 <class 'tensorflow.python.framework.ops.EagerTensor'> (2, 896, 1280, 3)
sample[1], i.e. labels:
 <class 'tensorflow.python.framework.ops.EagerTensor'> (2, 214830, 5)
```



