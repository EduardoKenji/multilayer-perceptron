# multilayer-perceptron

This was a project developed for Artificial Intelligence classes during my Bachelor of Science in Information Systems at Universidade de São Paulo. My own implementation of the Multilayer Perceptron, with one hidden layer and an output layer with 3 units, to correctly classify some handwritten uppercase letters (Z, X and S). I have recently refactored and improved the code.

* My own implementation of the Multilayer Perceptron. Currently with 576 input units, one hidden layer (15 neurons) and an output layer (3 units), to correctly classify the handwritten uppercase letters: Z, X and S.
* The learning rate was defaulted to 0.1, and the weights and bias are randomized between -0.5 and 0.5.
* The project was developed in Python.
* (Outside the commited project) The image of a letter (.PNG format from /unprocessed_images/) was converted into a Histogram of Oriented Gradients (from scikit-image.feature.hog).
* The neural network has 800 images, per letter, for the training set and 200 images, per letter, for the validating set.

<img src="pictures/multilayer_perceptron.PNG" width="500">
