import random
import math

class Neuron:
	def __init__(self, next_layer_size, expected_result):
		# Current value
		self.value = 0;
		# Neuron sum before evaluating, important feature to compute errors for backpropagating
		self.sum = 0;
		# Randomize bias between -0.5 and 0.5
		self.bias = random.random()-0.5
		# Randomize weights between -0.5 and 0.5
		self.weights = []
		for i in range(0, next_layer_size):
			self.weights.append(random.random()-0.5)
		# Expected output (not None only for output layer)
		self.expected_result = expected_result

class MultilayerPerceptron:
	# Default learning rate, number of layers and tuple with the amount of units/neurons per layer
	def __init__(self, learning_rate, units_per_layer_tuple, expected_output):
		self.learning_rate = learning_rate
		# Multilayer perceptron 2D weight matrix
		self.weight_matrix = []
		# A unit/neuron from the previous layer is connected to all units/neurons of the next layer (each connection is a weight)
		# Also, a unit/neuron from the next layer is connected to all units/neurons of the previous layer (each connection is a weight)
		for i in range(len(units_per_layer_tuple)):
			# Auxiliar list that represents the neurons of a layer and will be appended multilayer perceptron 2D weight matrix
			layer = []
			for j in range(units_per_layer_tuple[i]):
				# There at least one layer below the input or the output layer
				if(i != len(units_per_layer_tuple) - 1): 
					layer.append(Neuron(units_per_layer_tuple[i+1], None))
				# There is no layer below the output layer
				# We configure each output neuron accordingly with the needed output from expected_output
				else: 
					layer.append(Neuron(0, expected_output[j]))
			self.weight_matrix.append(layer)

	# Introduce feature vector to input layer
	def introduce_data_to_input_layer(self, feature_vector):
		for i in range(len(feature_vector)):
			self.weight_matrix[0][i].value = feature_vector[i]

	# Transfer data from the current layer to the next layer
	def transfer_data_to_next_layer(self, next_layer_id):	
		# Iterate cuurrent layer units
		for i in range(0, len(self.weight_matrix[next_layer_id])):
			neuron_sum = 0
			# A unit/neuron from the previous layer is connected to all units/neurons of the next layer (each connection is a weight)
			# Also, a unit/neuron from the next layer is connected to all units/neurons of the previous layer (each connection is a weight)
			# Iterate previous layer units
			for j in range(0, len(self.weight_matrix[next_layer_id-1])):
				neuron_sum += self.weight_matrix[next_layer_id-1][j].weights[i] * self.weight_matrix[next_layer_id-1][j].value
			neuron_sum += self.weight_matrix[next_layer_id][i].bias
			# Stores the neuron sum before the evaluation
			self.weight_matrix[next_layer_id][i].sum = neuron_sum
			# Neuron function will evaluate the its sum with a defined function
			evaluate_output = self.feedforward_evaluation(neuron_sum)
			self.weight_matrix[next_layer_id][i].value = evaluate_output

	# The feed-forward currently used function is the logistic function
	def feedforward_evaluation(self, value):
		return 1/(1 + math.exp(-value))

	# Derivate logistic to compute errors
	def derivate_logistic(self, value):
		return math.exp(value)/((math.exp(value)+1)**2)

	# Compute error by calculating offset from expected value (a letter, in this case) and adjust weights via backpropagation
	# Also returns total error from output units
	def compute_error_and_adjust_weights(self, expected_value):
		k_error_list, output_error_list = self.compute_error_from_output_layer(expected_value)
		# The output layer id (index) would be 2, so our multilayer perceptron, with one hidden layer, has its hidden layer id as 1
		j_error_list = self.compute_error_from_previous_layer(k_error_list, 1)
		# Adjust weights between output layer and the hidden layer
		self.adjust_weights(k_error_list, 2)
		# Adjust weights between hidden layer and the input layer
		self.adjust_weights(j_error_list, 1)
		# Returns total error from output units
		return output_error_list


	# Adjust weights between layer_index and (layer_index-1)
	def adjust_weights(self, error_list, layer_index):
		# Adjust weights between layer_index and (layer_index-1)
		for i in range(len(self.weight_matrix[layer_index-1])):
			for j in range(len(self.weight_matrix[layer_index])):
				# Increment weights with gradients
				self.weight_matrix[layer_index-1][i].weights[j] += self.learning_rate * self.weight_matrix[layer_index-1][i].value * error_list[j]
		# Adjust biases between layer_index and (layer_index-1)
		for i in range(len(self.weight_matrix[layer_index])):
			# Increment biases with gradients
			self.weight_matrix[layer_index][i].bias += self.learning_rate * error_list[i]

	# Compute error from hidden layers
	# k_error_list would be the list of errors for each neuron that came up from the lower/next layer
	def compute_error_from_previous_layer(self, k_error_list, previous_layer_id):
		j_error_list = []
		# Iterate neurons in previous_layer_id
		for i in range(len(self.weight_matrix[previous_layer_id])):
			neuron_error_sum = 0
			# Iterate neurons in previous_layer_id + 1
			for j in range(len(self.weight_matrix[previous_layer_id+1])):
				neuron_error_sum += self.weight_matrix[previous_layer_id][i].weights[j] * k_error_list[j]
			# Neuron error mount = neuron error sum * derivate neuron sum
			neuron_error_amount = neuron_error_sum * self.derivate_logistic(self.weight_matrix[previous_layer_id][i].sum)
			j_error_list.append(neuron_error_amount)
		return j_error_list

	# Compute error from output layer and return error per output unit
	def compute_error_from_output_layer(self, expected_value):
		k_error_list = []
		output_error_list = []
		output_layer_index = len(self.weight_matrix)-1
		# This value will correspond to each output unit expected output
		expected_output_from_neuron = 0
		# Amount of error per output unit
		error_amount = 0
		# Iterate output layer units
		for i in range(len(self.weight_matrix[output_layer_index])):
			# Compare if the the neuron was supposed or not to identify the letter from expected_value
			if(expected_value == self.weight_matrix[output_layer_index][i].expected_result):
				expected_output_from_neuron = 1
			else:
				expected_output_from_neuron = 0
			# Error amount = (expected value from output unit - actual value from output unit) * derivate neuron sum
			output_neuron_error = (expected_output_from_neuron - self.weight_matrix[output_layer_index][i].value)
			error_amount =  output_neuron_error * self.derivate_logistic(self.weight_matrix[output_layer_index][i].sum)
			k_error_list.append(error_amount)
			output_error_list.append(output_neuron_error)
		# Return list with all the error from output units
		return k_error_list, output_error_list

	# Return a dictionary with the letters as keys (ex.: "Z") and its correspondent output neuron value (from 0 to 1)
	def predict(self, feature_vector):
		# Introduce the feature vector to the input layer
		self.introduce_data_to_input_layer(feature_vector)
		# Transfer from input to hidden layers, and eventually from hidden layer to the output layer
		next_layer_id = 1
		self.transfer_data_to_next_layer(next_layer_id)
		next_layer_id = 2
		self.transfer_data_to_next_layer(next_layer_id)
		predictions = {}
		output_layer_index = len(self.weight_matrix)-1
		for i in range(len(self.weight_matrix[output_layer_index])):
			predictions[self.weight_matrix[output_layer_index][i].expected_result] = self.weight_matrix[output_layer_index][i].value
		return predictions