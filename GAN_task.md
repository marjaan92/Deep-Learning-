# Report: Generative Adversarial Network (GAN) for Image Generation

## Project description : 
The goal of the project is to reproduce the results to gain experience in how the generator is building the images with the
same qualities as the input MNIST dataset and trying to make the discriminator fool.

## Introduction

Generative models in artificial intelligence aim to generate new data that resembles the original dataset it was trained on. Techniques such as Generative Adversarial Networks (GANs) or Variational Autoencoders (VAEs) are popular approaches used for generating synthetic content. These models learn the underlying patterns and structures in the training data and generate new instances that resemble the original data distribution. Here we have implemented GANs.


## Choice of Generative Model:
Generative Adversarial Network (GAN) was selected due to its ability to generate realistic synthetic data by simultaneously training two neural networks: a generator and a discriminator.
GANs have shown remarkable performance in generating images, leveraging the adversarial training framework to produce high-quality outputs.

## Training Process:

#### Dataset: 
Utilized the MNIST dataset consisting of 28x28 grayscale handwritten digit images (0 to 9).

#### Data Preparation: 
Imported libraries such as NumPy, Pandas, Matplotlib, TensorFlow, and specific modules from the TensorFlow library required for building a neural network model.
Flattened the images to a 2D format and scaled pixel values to a range of (-1, +1).
Scaling Input Data: Scaled the pixel values of the images to the range (-1, +1) to prepare the data for better training. The original pixel values range from 0 to 255, and the scaling converts them to the range (-1, +1) by applying the formula x_train / 255.0 * 2 - 1 and x_test / 255.0 * 2 - 1.


#### Model Architecture:
1. Building the Generator Model:
Defines a generator model for a Generative Adversarial Network (GAN).latent_dim = 100 sets the dimensionality of the input noise vector for the generator. build_generator function constructs the generator model architecture. The generator consists of fully connected (Dense) layers with LeakyReLU activation and BatchNormalization between layers.
The output layer uses a tanh activation function to produce the generated image data.

2. Building the Discriminator Model:
Defines a discriminator model for the GAN.build_discriminator function constructs the discriminator model architecture.The discriminator also consists of fully connected (Dense) layers with LeakyReLU activation.
The output layer uses a sigmoid activation function to produce a binary classification output (real or fake).

#### Training Procedure:
Alternated between training the discriminator and the generator in a loop.
Discriminator learned to differentiate between real and fake images.
Generator learned to generate images that could potentially fool the discriminator.
Applied binary cross-entropy loss and Adam optimizer for training.


## Assessment of Output Quality:
#### Sample Generation:
Generated sample images at specific epochs to visualize the progress of image generation.
Inspected generated images at various epochs to observe the evolution of image quality.


#### Evaluation Metrics:
Evaluated discriminator and generator losses during training to assess convergence and stability.
Monitored discriminator and generator accuracies to gauge the model's ability to distinguish real and fake images.

#### Visual Inspection:
Plotted generated images at different epochs to assess their visual quality and resemblance to the original dataset.
Compared generated images against real MNIST digits for similarities and realistic features.
The code aims to plot the generated images saved at different epochs during the GAN training process. It utilizes the imread function from skimage.io to read and display the saved images.
Here, we are tried to display the generated images saved at 0 , 100 and 10000 epochs. Ensured that the file paths are 'GAN_Images/0.png', 'GAN_Images/1000.png' 'GAN_Images/10000.png' correctly pointed to the saved image files from the GAN training process. We can even adjust the file paths as needed to match the location where the images were saved.  


## Results and Observations:

Convergence: Observed the convergence of discriminator and generator losses over epochs.
Image Quality: Initially, the generated images might exhibit noise and lack clarity. However, over epochs, the images tend to become more structured and resemble handwritten digits.
Progression: Noticed improvements in image quality and resemblance to MNIST digits as training progressed.
Challenges: Some challenges might include mode collapse, where the generator produces limited variations or scenarios where the generator fails to fool the discriminator adequately.

## Conclusion:
At 0 epoch, There is no information that can be extracted from the generator, whereas the discriminator is capable enough to identify the fake case. At 1000 epochs, the generator is gradually extracting some information. However, it is not good enough to fool the discriminator. But the image generated at 10000 epochs images has shown improvements in quality compared to earlier epochs (e.g: 1000 epochs). There is visible progress in the generator's ability to replicate the qualities of the MNIST dataset. Also we have seen that at 10000 epochs, the generator has built the images with the same qualities as the MNIST dataset, and there are healthy chances of making the discriminator fool.









