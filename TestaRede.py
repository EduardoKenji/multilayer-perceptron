import pickle
import math
from MultilayerPerceptron import MultilayerPerceptron, Neuron, evaluate, evaluate2
import numpy as np
from skimage.io import imread, imshow
from skimage.feature import hog

def main():
	path = "Hog/entrada = 576, escondida = 10, saida = 3, alpha = 0.01/modelo 1/model.dat"
	path2 = "Hog/entrada = 576, escondida = 10, saida = 3, alpha = 0.5/modelo 2/model.dat"
	path3 = "Hog/entrada = 576, escondida = 10, saida = 3, alpha = 0.1/modelo 1/model.dat"
	pickle_in = open(path, "rb")
	mlp = pickle.load(pickle_in)
	testaRede(mlp)
	#testaImagem(mlp, "X4.png")

def testaImagem(mlp, path):
	A = imread(path)
	a1 = A[:,:,0]
	x, y = hog(a1, orientations = 9, pixels_per_cell = (16, 16), cells_per_block = (1, 1), visualise =  True)
	for i in range(len(x)):
		mlp.neurons[0][i].value = x[i]
	for i in range(0, len(mlp.neurons[1])):
		totalSum = 0
		for j in range(0, len(mlp.neurons[0])):
			totalSum += mlp.neurons[0][j].weights[i] * mlp.neurons[0][j].value
		totalSum += mlp.neurons[1][i].bias
		result = evaluate2(totalSum)
		mlp.neurons[1][i].value = result
	for i in range(0, len(mlp.neurons[2])):
		totalSum = 0
		for j in range(0, len(mlp.neurons[1])):
			totalSum += mlp.neurons[1][j].weights[i] * mlp.neurons[1][j].value
		totalSum += mlp.neurons[2][i].bias
		result = evaluate2(totalSum)
		mlp.neurons[2][i].value = result
	resposta = "Nao foi possivel definir"
	if(mlp.neurons[2][0].value > mlp.neurons[2][1].value and mlp.neurons[2][0].value > mlp.neurons[2][2].value):
		if(mlp.neurons[2][0].value > 0.8):
			resposta = "Z"
	elif(mlp.neurons[2][1].value > mlp.neurons[2][0].value and mlp.neurons[2][1].value > mlp.neurons[2][2].value):				
		if(mlp.neurons[2][1].value > 0.8):		
			resposta = "S"
	elif(mlp.neurons[2][2].value > mlp.neurons[2][0].value and mlp.neurons[2][2].value > mlp.neurons[2][1].value):
		if(mlp.neurons[2][2].value > 0.8):
			resposta = "X"
	print(resposta)
	
def testaRede(mlp):
	respostasZ = 0
	respostasS = 0
	respostasX = 0
	erroTotal = 0.0
	for l in range(0, 900):
		if(l%3 == 0): charAtual = 'a'
		elif(l%3 == 1): charAtual = '3'
		else: charAtual = '8'
		trueFileNum = str(math.floor(l/3)+1000)
		fileName = "dataset1/testes/train_5"+charAtual+"_0"+trueFileNum+".png"
		A = imread(fileName)
		a1 = A[:,:,0]
		x, y = hog(a1, orientations = 9, pixels_per_cell = (16, 16), cells_per_block = (1, 1), visualise =  True)
		for i in range(len(x)):
			mlp.neurons[0][i].value = x[i]
		for i in range(0, len(mlp.neurons[1])):
			totalSum = 0
			for j in range(0, len(mlp.neurons[0])):
				totalSum += mlp.neurons[0][j].weights[i] * mlp.neurons[0][j].value
			totalSum += mlp.neurons[1][i].bias
			result = evaluate2(totalSum)
			mlp.neurons[1][i].value = result
		for i in range(0, len(mlp.neurons[2])):
			totalSum = 0
			for j in range(0, len(mlp.neurons[1])):
				totalSum += mlp.neurons[1][j].weights[i] * mlp.neurons[1][j].value
			totalSum += mlp.neurons[2][i].bias
			result = evaluate2(totalSum)
			mlp.neurons[2][i].value = result
		erroTotalDaImagem = 0
		erro = 0
		expectedIndex = 0
		if(charAtual == 'a'): expectedIndex = 0
		elif(charAtual == '3'): expectedIndex = 1
		else: expectedIndex = 2
		expected = 0
		for i in range(0, len(mlp.neurons[2])):
			if(i == expectedIndex): expected = 1
			else: expected = 0
			erro = expected - mlp.neurons[2][i].value
			erro = erro * erro
			erro = erro/2
			erroTotalDaImagem += erro
		erroMedioDaImagemPorNeuronio = erroTotalDaImagem/3	
		erroTotal += erroMedioDaImagemPorNeuronio
		resposta = "Nao foi possivel definir"
		if(mlp.neurons[2][0].value > mlp.neurons[2][1].value and mlp.neurons[2][0].value > mlp.neurons[2][2].value):
			if(mlp.neurons[2][0].value > 0.8):
				resposta = "Z"
		elif(mlp.neurons[2][1].value > mlp.neurons[2][0].value and mlp.neurons[2][1].value > mlp.neurons[2][2].value):				
			if(mlp.neurons[2][1].value > 0.8):		
				resposta = "S"
		elif(mlp.neurons[2][2].value > mlp.neurons[2][0].value and mlp.neurons[2][2].value > mlp.neurons[2][1].value):
			if(mlp.neurons[2][2].value > 0.8):
				resposta = "X"
		if(charAtual == 'a' and resposta == "Z"): respostasZ = respostasZ + 1
		if(charAtual == '3' and resposta == "S"): respostasS = respostasS + 1
		if(charAtual == '8' and resposta == "X"): respostasX = respostasX + 1
	print("["+str(len(mlp.neurons[0]))+", "+str(len(mlp.neurons[1]))+", "+str(len(mlp.neurons[2]))+"], alpha = "+str(mlp.learningRate))
	print("Erro médio por imagem de cada neurônio: "+str(erroTotal/900))
	print("Letras Z: "+str(respostasZ)+"/300; Letras S: "+str(respostasS)+"/300; Letras X: "+str(respostasX)+"/300; Total: "+str(respostasZ + respostasS + respostasX)+"/900")

if __name__ == "__main__": 
	main()
