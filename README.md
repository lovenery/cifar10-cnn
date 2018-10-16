# DTV

## Build

```bash
pip install keras tensorflow matplotlib h5py Pillow
```

## CIFAR-10

- https://www.cs.toronto.edu/~kriz/cifar.html
- https://en.wikipedia.org/wiki/CIFAR-10
- 60000 32x32 colour images (RGB)
    - 10 classes
    - 6000 images per class
    - 50000 training images and 10000 test images
- five training batches and one test batch
    - each batch with 10000 images
- The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.
    - label: 0~9

## Refs

- load dataset
    - https://keras.io/datasets/#cifar10-small-image-classification
    - https://github.com/keras-team/keras/blob/master/keras/datasets/cifar10.py
    - https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/python/keras/datasets/cifar10.py
- http://yhhuang1966.blogspot.com/2018/04/keras-cifar-10.html
- http://yhhuang1966.blogspot.com/2018/04/keras-cnn-cifar-10.html
- https://ithelp.ithome.com.tw/articles/10192162
- https://blog.plon.io/tutorials/cifar-10-classification-using-keras-tutorial/
- https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
- https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py

## Notes

- https://towardsdatascience.com/deep-learning-framework-power-scores-2018-23607ddf297a
