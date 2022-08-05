![CI Tests](https://github.com/soerenberg/variational-autoencoder/actions/workflows/ci_tests.yml/badge.svg)
[![Build Status](https://app.travis-ci.com/soerenberg/variational-autoencoder.svg?token=hpcdeWX5ho5Gtj7Nsa3k&branch=main)](https://app.travis-ci.com/soerenberg/variational-autoencoder)
[![codecov](https://codecov.io/gh/soerenberg/variational-autoencoder/branch/main/graph/badge.svg?token=0NPJ3SKUTK)](https://codecov.io/gh/soerenberg/variational-autoencoder)

# Variational Autoencoder

Implementation of a Variational Autoencoder model applied to several datasets.

The model is implemented in `TensorFlow 2` using its `Keras` API `tf.keras`.
The primary goal of this project was not to find the most performant ANN
architecture for a general VAE or even to find very well-tuned hyperparameters.
The primary goal wast to have a small, maintanable, reliable & well-tested code
which is easy to extend, and present some simple results to demonstrate that
the model is quite able to create new images instead of just reconstructing
training instances.

Below you find results for some of the most well-known datasets.


## Results

### MNIST dataset

#### Latent dimension 25

The following animation shows how the decoder of the VAE model creates images
for a set of 16 points randomly sampled in the latent space held fixed over
all epochs.

![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/grid_animations.gif?raw=true "Training progress of VAE")

Generated | Training set
--- | ---
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/image_00_step_00000097.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/train_image_closest_to_0_label_0.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/image_01_step_00000097.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/train_image_closest_to_1_label_6.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/image_02_step_00000097.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/train_image_closest_to_2_label_7.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/image_03_step_00000097.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/train_image_closest_to_3_label_5.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/image_04_step_00000097.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/train_image_closest_to_4_label_8.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/image_05_step_00000097.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/train_image_closest_to_5_label_6.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/image_06_step_00000097.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/train_image_closest_to_6_label_9.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/image_07_step_00000097.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/train_image_closest_to_7_label_3.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/image_08_step_00000097.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/train_image_closest_to_8_label_2.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/image_09_step_00000097.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/train_image_closest_to_9_label_0.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/image_10_step_00000097.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/train_image_closest_to_10_label_6.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/image_11_step_00000097.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/train_image_closest_to_11_label_8.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/image_12_step_00000097.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/train_image_closest_to_12_label_5.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/image_13_step_00000097.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/train_image_closest_to_13_label_3.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/image_14_step_00000097.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/train_image_closest_to_14_label_3.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/image_15_step_00000097.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/mnist_latent_dim_25/train_image_closest_to_15_label_6.png?raw=true "Image from training set")


### Fashion-MNIST dataset

#### Latent dimension 100

The following animation shows how the decoder of the VAE model creates images
for a set of 16 points randomly sampled in the latent space held fixed over
all epochs.

![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/grid_animations.gif?raw=true "Training progress of VAE")

Generated | Training set
--- | ---
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/image_00_step_00000041.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/train_image_closest_to_0_label_8.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/image_01_step_00000041.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/train_image_closest_to_1_label_3.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/image_02_step_00000041.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/train_image_closest_to_2_label_2.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/image_03_step_00000041.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/train_image_closest_to_3_label_8.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/image_04_step_00000041.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/train_image_closest_to_4_label_9.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/image_05_step_00000041.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/train_image_closest_to_5_label_2.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/image_06_step_00000041.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/train_image_closest_to_6_label_6.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/image_07_step_00000041.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/train_image_closest_to_7_label_0.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/image_08_step_00000041.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/train_image_closest_to_8_label_1.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/image_09_step_00000041.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/train_image_closest_to_9_label_8.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/image_10_step_00000041.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/train_image_closest_to_10_label_3.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/image_11_step_00000041.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/train_image_closest_to_11_label_9.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/image_12_step_00000041.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/train_image_closest_to_12_label_2.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/image_13_step_00000041.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/train_image_closest_to_13_label_5.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/image_14_step_00000041.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/train_image_closest_to_14_label_6.png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/image_15_step_00000041.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/train_image_closest_to_15_label_6.png?raw=true "Image from training set")

#### Visualization of 10k points of the train set in the latent space

The points from the 100 dimensional latent space have been projected into the
plane using PCA.

![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/planar_encoding_train_step_00000041.png?raw=true "Encoding to latent space")


#### Visualization of 10k points of the test set in the latent space

The points from the 100 dimensional latent space have been projected into the
plane using PCA.

![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/fashion_mnist_latent_dim_100/planar_encoding_test_step_00000041.png?raw=true "Encoding to latent space")

### CIFAR-10 dataset

#### Latent dimension 100

The following animation shows how the decoder of the VAE model creates images
for a set of 16 points randomly sampled in the latent space held fixed over
all epochs.

![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/grid_animations.gif?raw=true "Training progress of VAE")

Generated | Training set
--- | ---
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/image_00_step_00000044.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/train_image_closest_to_0_label_[0].png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/image_01_step_00000044.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/train_image_closest_to_1_label_[2].png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/image_02_step_00000044.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/train_image_closest_to_2_label_[6].png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/image_03_step_00000044.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/train_image_closest_to_3_label_[4].png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/image_04_step_00000044.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/train_image_closest_to_4_label_[8].png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/image_05_step_00000044.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/train_image_closest_to_5_label_[2].png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/image_06_step_00000044.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/train_image_closest_to_6_label_[0].png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/image_07_step_00000044.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/train_image_closest_to_7_label_[2].png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/image_08_step_00000044.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/train_image_closest_to_8_label_[0].png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/image_09_step_00000044.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/train_image_closest_to_9_label_[6].png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/image_10_step_00000044.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/train_image_closest_to_10_label_[0].png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/image_11_step_00000044.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/train_image_closest_to_11_label_[4].png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/image_12_step_00000044.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/train_image_closest_to_12_label_[6].png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/image_13_step_00000044.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/train_image_closest_to_13_label_[5].png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/image_14_step_00000044.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/train_image_closest_to_14_label_[3].png?raw=true "Image from training set")
![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/image_15_step_00000044.png?raw=true "Image generated from VAE") | ![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/train_image_closest_to_15_label_[3].png?raw=true "Image from training set")

#### Visualization of 10k points of the train set in the latent space

The points from the 100 dimensional latent space have been projected into the
plane using PCA.

![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/planar_encoding_train_step_00000044.png?raw=true "Encoding to latent space")


#### Visualization of 10k points of the test set in the latent space

The points from the 100 dimensional latent space have been projected into the
plane using PCA.

![alt text](https://github.com/soerenberg/variational-autoencoder/blob/main/images/cifar10_latent_dim_100/planar_encoding_test_step_00000044.png?raw=true "Encoding to latent space")
