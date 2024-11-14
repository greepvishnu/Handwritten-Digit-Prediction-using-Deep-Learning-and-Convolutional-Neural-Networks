# Handwritten-Digit-Prediction-using-Deep-Learning-and-Convolutional-Neural-Networks
The objective of this project is to predict handwritten digits using Deep Learning Neural Networks (DNNs) and Convolutional Neural Networks (CNNs), and to compare the accuracy of both models using the MNIST dataset.
## Approach:
To solve the problem of handwritten digit classification, we will use two approaches:

# Deep Neural Networks (DNNs): A traditional neural network architecture where each layer is fully connected.
Convolutional Neural Networks (CNNs): A more advanced architecture specifically designed for image data, which uses convolutional layers to detect patterns in the images.

## Python Libraries Used:
The following Python libraries were used to implement the models:

Keras: A high-level neural networks API written in Python, running on top of TensorFlow. It simplifies the creation and training of deep learning models.

TensorFlow: A popular open-source library for machine learning, which is the backend for Keras and provides tools for training and deploying models.

NumPy: A library used for numerical operations in Python. It is used for manipulating large arrays and matrices of numeric data.

Matplotlib: A plotting library for Python. It is used to visualize data such as images, model training progress, and results.

##Dataset: MNIST Dataset

The MNIST dataset (Modified National Institute of Standards and Technology) is a well-known dataset that contains 70,000 images of handwritten digits (0–9). The dataset is split into:

60,000 images for training.
10,000 images for testing. Each image is a 28x28 pixel grayscale image, and each image is labeled with a corresponding digit (0–9). The MNIST dataset is often used as a benchmark in machine learning for digit recognition tasks.

##Steps Followed:

Step 1: Loading the Necessary Libraries and Datasets
In this step, we import all the required libraries and load the MNIST dataset.

Step 2: Splitting the Training and Testing Data
The MNIST dataset is already split into training and testing sets, but it’s always good practice to explicitly understand how the data is structured.

x_train: Contains the 60,000 training images.
y_train: Contains the labels for the training images.
x_test: Contains the 10,000 testing images.
y_test: Contains the labels for the testing images.

Step 3: Checking Their Shapes and Visualizing Some Examples
Before proceeding with model creation, we check the shape of the data and visualize some samples to understand what we're working with.

The shape of x_train will be (60000, 28, 28), where 60000 is the number of training samples, and each image is 28x28.
The shape of y_train will be (60000,), where each value is a digit (0-9).

Step 4: Flattening the Data to 1D for Deep Neural Networks (DNN)
Deep Neural Networks (DNNs) typically expect the input data to be a 1D vector. Therefore, we flatten the 28x28 images into 1D vectors of 784 features (28 * 28 = 784) to feed into the model.

Flattening: The images are reshaped from (28, 28) to (784,) to create a 1D vector of pixel values.
Normalization: Pixel values are normalized from the range [0, 255] to [0, 1] by dividing by 255.

Step 5: Standardizing the Data to Improve Accuracy
Standardizing the data is a common practice to help with model convergence during training. Here, we normalize the pixel values to the range [0, 1], which was already done in the previous step.

If you wanted to standardize the data to have zero mean and unit variance, you could apply a different transformation, but normalization is a simpler and often effective approach for image data.

Step 6: Creating a Deep Neural Network (DNN) with 100 Hidden Units
For the Deep Neural Network model, we define a simple architecture with one hidden layer containing 100 units.

Hidden Layer: The first layer has 100 units, with the ReLU activation function.
Output Layer: The final layer has 10 units (one for each digit), using the softmax activation to output probabilities for each class (digit 0-9).
Optimizer: Adam optimizer is used for efficient gradient-based optimization.
Loss Function: sparse_categorical_crossentropy is used as the loss function, which is suitable for multi-class classification with integer labels.

Step 7: Activation Functions Used

ReLU (Rectified Linear Unit): The activation function used in the hidden layers. ReLU is a popular choice because it helps with non-linearity and avoids the vanishing gradient problem.
Softmax: The activation function used in the output layer to produce a probability distribution over the 10 possible classes (digits 0-9).

Step 8: Optimizer and Loss Function

Optimizer: Adam: The Adam optimizer is used due to its ability to adapt the learning rate during training, making it a good choice for most neural networks.
Loss: Sparse Categorical Cross-Entropy: This loss function is used for multi-class classification where the labels are integers. It measures the difference between the predicted probabilities and the actual class labels.

Step 9: Training the Deep Neural Network Model
In this step, we train the Deep Neural Network (DNN) model using the training dataset for 20 epochs and evaluate its performance.

Epochs: The model is trained for 20 epochs, which means the entire dataset will be passed through the model 20 times during training.
Batch Size: A batch size of 64 means the model processes 64 images at a time before updating the weights.
Validation Data: We use the test dataset (x_test_flat, y_test) to validate the model's performance during training and prevent overfitting.

## Training Results:
Training Accuracy: After training for 20 epochs, the model achieved an accuracy of 99% on the training data.
Test Accuracy: The model's performance on the test data was slightly lower, with an accuracy of 97%.
This shows that the model has learned well, but there might be slight overfitting since the training accuracy is higher than the test accuracy. However, these are still very high numbers, indicating good generalization.

Step 10: Predicting the Digit Dataset Using Convolutional Neural Network (CNN)
To improve the performance of our model, we transition to using a Convolutional Neural Network (CNN), which is typically better suited for image data. In this step, we repeat the previous steps but apply the CNN instead of the DNN.

Repeat Steps 1 to 3: We begin by repeating the first three steps of loading the necessary libraries and visualizing the dataset. However, instead of flattening the dataset for input into a DNN, we will feed the data directly into the CNN.

Step 11: Convolutional Neural Network Architecture
The CNN architecture used in this project includes two convolutional layers and max-pooling layers.

1st Convolutional Layer:

Filters: 40 filters are used, which can detect 40 different features from the input images.
Kernel Size: A 3x3 kernel is used to slide over the input images and extract features.
Activation Function: The ReLU activation function is applied to introduce non-linearity.
1st Max-Pooling Layer:

Pool Size: 2x2 pool size, which reduces the spatial dimensions of the feature maps.
2nd Convolutional Layer:

Filters: 30 filters, further reducing the dimensionality of the extracted features.
Kernel Size: Again, a 3x3 kernel is used for feature extraction.
Activation Function: ReLU activation.
2nd Max-Pooling Layer:

A second 2x2 max-pooling layer follows the second convolutional layer to further downsample the feature maps.
Flattening Layer:

After the convolutional and pooling layers, we flatten the multi-dimensional data into a 1D vector to input into the fully connected (dense) layers.
Fully Connected Layers:

Dense Layer: A fully connected layer with 100 units and ReLU activation.
Output Layer: A softmax output layer with 10 units corresponding to the 10 digits (0-9).

Step 12: Training the Convolutional Neural Network (CNN) Model
We train the CNN model using the same process as we did for the DNN model. The primary difference here is that we do not flatten the images before passing them to the model. Instead, we input the raw 28x28 images directly into the CNN, which will automatically learn the features through the convolutional layers.

Epochs: 20 epochs, similar to the DNN model.
Batch Size: 64 samples at a time.
Validation Data: Using the test dataset for validation during training.
Training Results:
Training Accuracy: The CNN model achieved a training accuracy of 97%.
Test Accuracy: The CNN model showed a test accuracy of 98%, which is even higher than its performance on the training data.
This suggests that the CNN is generalizing well and is able to achieve slightly better accuracy on the test data compared to the DNN. CNNs often outperform fully connected DNNs on image classification tasks because they are specifically designed to detect spatial hierarchies and patterns in images.

Step 13: Comparing the Performance of DNN and CNN
DNN Results:

Training Accuracy: 99%
Test Accuracy: 97%
CNN Results:

Training Accuracy: 97%
Test Accuracy: 98%
Analysis:

# DNN:

The DNN performed very well with a 99% accuracy on the training data, but it slightly overfit, as evidenced by the 97% test accuracy.
The DNN is more prone to overfitting when applied to image data because it doesn’t leverage spatial relationships in the data as well as CNNs do.

# CNN:

The CNN model showed slightly better test accuracy (98%) than the DNN (97%), despite having a lower training accuracy (97% compared to 99%). This suggests the CNN is better at generalizing to unseen data and not overfitting, as it uses convolutional and pooling layers to better capture spatial features from the images.
CNNs are typically more effective for image-related tasks, and their ability to learn local features (edges, textures, shapes) helps improve performance on tasks like digit classification.

## Conclusion:
From the results, we can conclude:

The CNN model outperforms the DNN model in terms of test accuracy (98% vs. 97%).
CNNs are a better choice for image classification tasks like the MNIST digit dataset due to their ability to capture local patterns and reduce overfitting.




