# multilayer-perceptron

This was a project developed for Artificial Intelligence classes during my Bachelor of Science in Information Systems at Universidade de São Paulo. A raw implementation of the Multilayer Perceptron, with one hidden layer and an output layer with 3 units, to correctly classify some handwritten uppercase letters (Z, X and S).

* My own implementation of the Multilayer Perceptron. Currently with 576 input units, one hidden layer (15 neurons) and an output layer (3 units), to correctly classify the handwritten uppercase letters: Z, X and S.
* The project was developed in Python and supported by scikit-image library.
* (Outside the commited project) The image of a letter (.PNG format from /unprocessed_images/) was converted into a Histogram of Oriented Gradients (from scikit-image.feature.hog) and then used as input to the neural network.
* The neural network training is validated by the k-fold cross-validation procedure (800 images, per letter, for the training set and 200 images, per letter, for the validating set).
