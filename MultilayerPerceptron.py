import math
import random

class Neuron:
	def __init__(self, nextLayerSize):
		self.value = 0;
		#Define o bias
		self.bias = random.random()-0.5
		#Define e cria a lista de pesos
		self.weights = []
		for i in range(0, nextLayerSize):
			self.weights.append(random.random()-0.5)

class MultilayerPerceptron:
	def __init__(self, layerNum, learningRate):
		#Define a velocidade de aprendizado da rede
		self.learningRate = learningRate
		#Define e cria as camadas e seus respectivos neuronios em uma lista bidimensional de neuronios
		a = []
		b = []
		c = []
		for i in range(0, layerNum[0]):
			a.append(Neuron(layerNum[1]))
		for i in range(0, layerNum[1]):
			b.append(Neuron(layerNum[2]))
		for i in range(0, layerNum[2]):
			c.append(Neuron(0))
		self.neurons = [a, b, c]

def evaluate(value):
	return math.tanh(value)

def evaluate2(value):
	return 1/(1 + math.exp(-value))

#print(evaluate2(-1.5))
#neuron1 = Neuron(0.1, 0)
#print(neuron1.evaluate(-0.5))
#layerNum = [576, 10, 3]
#mlp = MultilayerPerceptron(layerNum, 0.1)
#print(repr(layerNum[0])+" "+repr(layerNum[2])+" "+repr(len(layerNum)))
#print(mlp.neurons[2][2].weights[9])
