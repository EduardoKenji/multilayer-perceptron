import random
import constant
import math
import pickle
import time
from multilayer_perceptron import MultilayerPerceptron

# The main function, the first called in the application
def main():
	# Default learning rate, tuple with the amount of units/neurons per layer
	mlp = MultilayerPerceptron(constant.STD_LEARNING_RATE, constant.STD_UNITS_PER_LAYER, constant.STD_EXPECTED_OUTPUT)
	# Get a list of tuples containing (feature vector, expected letter) for training
	training_data_list = create_training_data_list(constant.TRAINING_SET_RANGE)
	# Get a list of tuples containing (feature vector, expected letter) for validation
	validation_data_list = create_validation_data_list(constant.TRAINING_SET_RANGE, constant.VALIDATION_SET_RANGE)
	# Shuffle data
	random.shuffle(training_data_list)
	random.shuffle(validation_data_list)
	# Get model accuracy before training
	print()
	print("=== Model accuracy before training ===")
	get_model_accuracy_with_validation_set(mlp, validation_data_list)
	# Train model with available training data for constant.MAX_EPOCH
	mlp = train_mlp(mlp, training_data_list, constant.MAX_EPOCH)
	# Get model accuracy after training
	print("=== Model accuracy after training ===")
	get_model_accuracy_with_validation_set(mlp, validation_data_list)

# mlp: Multilayer Perceptron model
# training_data_list: a list of tuples containing (feature vector, expected letter) for training
# this function will train the model with the available training data from "processed_images/training" folder until max_epoch
def train_mlp(mlp, training_data_list, max_epoch):
	# Debug on console to know the training is starting
	print("=== Start of training ===")
	# Train models for, at maximum, max_epoch
	for i in range(max_epoch):
		# training_data_list[j][0]: Feature vector
		# training_data_list[j][1]: Expected letter for the feature vector
		# error_from_this_epoch: Error list from all units/neurons from the output layer during the current epoch
		error_list_from_this_epoch = []
		start = time.time()
		for j in range(len(training_data_list)):
			# Introduce the feature vector to the input layer
			mlp.introduce_data_to_input_layer(training_data_list[j][0])
			# Transfer from input to hidden layers, and eventually from hidden layer to the output layer
			next_layer_id = 1
			mlp.transfer_data_to_next_layer(next_layer_id)
			next_layer_id = 2
			mlp.transfer_data_to_next_layer(next_layer_id)
			# Compute error by calculating offset from expected value (a letter, in this case) and adjust weights via backpropagation
			# error_from_this_input: Error list from all units/neurons from the output layer during the current input
			error_from_this_input = mlp.compute_error_and_adjust_weights(training_data_list[j][1])
			error_list_from_this_epoch.append(error_from_this_input)
		# Debug on console to know the elapsed epoch, its elapsed time and the total error from it
		# This is the MSE from output units
		end = time.time()
		print("Epoch "+str(i)+", mean squared error: "+str(get_mse(error_list_from_this_epoch))+", elapsed time: {0:.2f}".format(end-start)+"s")
	print("=== End of training ===")
	return mlp

# Get mean squared error for an epoch
def get_mse(error_list_from_this_epoch):
	# 2400 training inputs per epoch * 3 output units = 7200 as total amount of errors
	total_amount_of_errors = len(error_list_from_this_epoch) * len(error_list_from_this_epoch[0])
	total_sqrd_error_sum = 0
	# MSE = (1/n) * (sum(error)^2)
	for i in range(len(error_list_from_this_epoch)):
		list_with_sqrd_elems = [x**2 for x in error_list_from_this_epoch[i]]
		total_sqrd_error_sum += sum(list_with_sqrd_elems)
	mse = total_sqrd_error_sum/total_amount_of_errors
	return mse

# Get model accuracy and debug print on console some useful information
def get_model_accuracy_with_validation_set(model, validation_data_list):
	correct_predictions = 0
	z_correct = 0
	x_correct = 0
	s_correct = 0
	number_of_images_per_letter = len(validation_data_list)//3
	for i in range(len(validation_data_list)):
		# validation_data_list[i][0]: Feature vector
		# validation_data_list[i][1]: Expected letter for the feature vector
		prediction = model.predict(validation_data_list[i][0])
		predicted_letter = max(prediction, key=lambda i: prediction[i])
		if(predicted_letter == validation_data_list[i][1]):
			correct_predictions += 1
			if(predicted_letter == "Z"):
				z_correct += 1
			if(predicted_letter == "X"):
				x_correct += 1
			if(predicted_letter == "S"):
				s_correct += 1
	accuracy = correct_predictions/(len(validation_data_list))
	# Debug on console to know the model accuracy and the number of correct predictions vs total amount of images per letter
	print("Z:"+str(z_correct)+"/"+str(number_of_images_per_letter)+
		" X:"+str(x_correct)+"/"+str(number_of_images_per_letter)+
		" S:"+str(s_correct)+"/"+str(number_of_images_per_letter)+
		" Total:"+str(s_correct+x_correct+z_correct)+"/"+str(len(validation_data_list)))
	print("Model accuracy: {0:.2f}".format(accuracy))
	return accuracy


# There are 600 images for validation (200 per letter)]
# training_set_range: Number of images per letter in training set
# validation_set_range: Number of images per letter in training set
# the validation images are number in the interval: [training_set_range, validation_set_range)
# Return a list of tuples containing (feature vector, expected letter) for validation
def create_validation_data_list(training_set_range, validation_set_range):
	validation_data_list = []
	for i in range(training_set_range, validation_set_range):
		validation_data_list.append(get_feature_vector_expected_tuple("processed_images/validation/z_"+process_number_to_string(i)+".txt", "Z"))
		validation_data_list.append(get_feature_vector_expected_tuple("processed_images/validation/x_"+process_number_to_string(i)+".txt", "X"))
		validation_data_list.append(get_feature_vector_expected_tuple("processed_images/validation/s_"+process_number_to_string(i)+".txt", "S"))
	# Return a list of tuples containing (feature vector, expected letter) for validation
	return validation_data_list

# There are 2400 images for training (800 per letter)
# training_set_range: Number of images per letter in training set
# Return a list of tuples containing (feature vector, expected letter) for training
def create_training_data_list(training_set_range):
	training_data_list = []
	for i in range(training_set_range):
		training_data_list.append(get_feature_vector_expected_tuple("processed_images/training/z_"+process_number_to_string(i)+".txt", "Z"))
		training_data_list.append(get_feature_vector_expected_tuple("processed_images/training/x_"+process_number_to_string(i)+".txt", "X"))
		training_data_list.append(get_feature_vector_expected_tuple("processed_images/training/s_"+process_number_to_string(i)+".txt", "S"))
	# Return a list of tuples containing (feature vector, expected letter) for training
	return training_data_list

# This function is used to to help building the name of the processed image for pickle to load
# Convert a integer (ex.: 1) to a string version with 0's to the left (ex.: "0001")
def process_number_to_string(num):
	# Convert i to a string called string_i
	string_num = str(num)
	# Until the number lenght
	while(len(string_num)<4):
		# Concatenate 0's to the left of the string_i
		string_num = "0" + string_num
	return string_num

# pickle_file_address: The processed image by scikit-learn.feature.hog serialized in a text file
# expected_value: The correct letter
# Return a tuple containing (feature vector, expected letter)
def get_feature_vector_expected_tuple(pickle_file_address, expected_value):
	pickle_in = open(pickle_file_address, "rb")
	feature_vector = pickle.load(pickle_in)
	return (feature_vector, expected_value)

if __name__ == "__main__": 
	main()
