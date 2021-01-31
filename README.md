# CIFAR-10 recognition with CNN

## CIFAR-10 dataset: 
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images.

- The test batch contains exactly 1000 randomly-selected images from each class.
- The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another.

Between them, the training batches contain exactly 5000 images from each class.

![cifar10](https://user-images.githubusercontent.com/69978820/106388793-6566eb80-63e0-11eb-916a-aacde8d57583.png)

## Load dataset 

```python 
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```
Visualize randomly some images 

```python
import matplotlib.pyplot as plt
import random

plt.figure(figsize = (12,5))
for i in range(8):
  ind = random.randint(0,len(X_train))
  plt.subplot(240 + 1 + i)
  plt.imshow(X_train[ind])

plt.show()
```
![visualize images](https://user-images.githubusercontent.com/69978820/106388944-00f85c00-63e1-11eb-9f33-39d8d8d5fdda.png)

## Create CNN model for recognizing images: 
### Summary: 
- The model is created with 3 hidden layers and one fully connected layer
- Methods to ovoid overfiting: 
  * Batch normalization
  * Dropout
  * Weight decay (weight regularization)
  * Data augmentation with rotation_range = 5 degree, width_shift_range = 0.1, height_shift_range = 0.1 and horizontal flip
- Optimization method: Stochastic Gradient Descent, with learning rate lr = 0.001
- Metric: accuracy
- The model is trained with batch size = 64 and 200 epoches 

### Results 

The accuracy on the validation is: 

## Recognize images on the test set:
```python
import random
Cats = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'struck']

plt.figure(figsize = (15,8))
for i in np.arange(8):
  ind = random.randint(0,len(X_test))
  img = X_test[ind]
  img = img.reshape(1,32,32,3)
  img = img.astype('float32')
  img = img/255.0
  v_p = model.predict_classes(img)
  plt.subplot(240+1+i)
  plt.imshow(X_test[ind])
  plt.title(Cats[v_p[0]])
  ```
  ![prediction](https://user-images.githubusercontent.com/69978820/106392166-f940b380-63f0-11eb-926a-997c533be3f8.png)

