# multilayer-perceptron
This was a project developed for Artificial Intelligence classes during my Bachelor of Science in Information Systems at Universidade de São Paulo. 
* A raw implementation of the Multilayer Perceptron, with one hidden layer and an output layer with 3 units, to correctly classify some handwritten uppercase letters (Z, X and S). 
* The project was developed in Python and supported by scikit-image library. 
* The image of a letter (.PNG format) is converted into a Histogram of Oriented Gradients (from scikit-image.feature.hog) and then used as input to the neural network. 
* The neural network training is validated by the k-fold cross-validation procedure (1000 images, per letter, for the training set and 300 images, per letter, for the validating set). 
* The variables names, modules names, classes names and comments are currently in Portuguese.
