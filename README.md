
# Artistic-Style-Transfer-NST
A deep learning project based on Convolutional Neural Network(CNN) using pyTorch.

Neural style transfer (NST) is a very neat idea. NST builds on the key idea that,
it is possible to separate the style representation and content representations in a CNN, learnt during a computer vision task (e.g. image recognition task) Following this concept, NST employs a pretrained convolution neural network (CNN) to transfer styles from a given image to another. This is done by defining a loss function that tries to minimise the differences between a content image, a style image and a generated image.

Steps followed are:

1. Defining the model architecture -
* Tried implementing my own mini VGGNET 16 on CIFAR100 dataset. But as my laptop's hardware is not strong, I couldn't train my model.
* Used a pre-implemented VGGNET 19 and extracted the information for the content and the style layer.

2. Training(on my own model)-
* I used the CIFAR-100 dataset. It has 100 classes with 600 images each. There are 500 training images and 100 testing images per class. 
* The train dataset: validation dataset is 9 : 1.
* Loss Functions used are:
  
a) **Content Loss**: Measures the difference between the content of the generated image and the content of the target image. It is usually the mean squared difference between the feature maps.

b) **Style Loss**: Measures the difference between the style of the generated image and the style of the target image. It is calculated as the mean squared difference of the Gram matrices.

3. Style Adaptation-
While training neural network on image dataset, all images are their mean removed, and values are rescaled according to the standard deviation of the dataset. This is a common processing which makes the neural network convergence easier. 

4. Evaluation:
We chose LBFGS as an optimisation algorithm for our gradient descent. 

## Dependency/ Library Used
torchvision

numpy

io 

matplotlib.pyplot

requests

**Note: Simply downloading and running the jupyter notebook will work. No extra dependencies need to be installed. Everything has been explained in a very detailed manner in the Jupyter Notebook. The limitations and the potential improvements are mentioned in the notebook itself. I would like to express my special thanks to some excellent blog writers on medium and other websites to explain NST in a very easy and simple manner.**
