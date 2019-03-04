import numpy as np
import random
import copy
import math
import pickle
import os
import errno
import time
# Minha implementação de Multilayer Perceptron
from MultilayerPerceptron import MultilayerPerceptron, Neuron, evaluate, evaluate2

def main():
	executa(576, 10, 3, 0.1, 1, 100)

def executa(rede_camada_de_entrada_neuronios, rede_camada_escondida_neuronios, rede_camada_de_saida_neuronios, rede_taxa_de_aprendizado, rede_min_epocas, rede_max_epocas):
	#print ("Execucao em "+time.strftime("%d/%m/%Y")+" "+time.strftime("%H:%M"))
	k = 5
	# Faz índices das imagens que serao acessadas de acordo com o k-fold cross-validation
	indices = kFoldCrossValidation(k, numImagem=3000, boolRandom=True)
	# Cria lista com o tamanho das 3 camadas do MLP
	layers = [rede_camada_de_entrada_neuronios, rede_camada_escondida_neuronios, rede_camada_de_saida_neuronios]
	# Cria k modelos com inicializacao de pesos aleatoria
	modelos = []
	for i in range(0, k):
		modelos.append(MultilayerPerceptron(layers, rede_taxa_de_aprendizado))
	# Treina os k modelos
	for i in range(0, len(modelos)):
		cria_diretorio(i+1, "Hog", modelos[i])		
		path = os.getcwd()+"/Hog/entrada = "+str(rede_camada_de_entrada_neuronios)+", escondida = "+str(rede_camada_escondida_neuronios)+", saida = "+str(rede_camada_de_saida_neuronios)+", alpha = "+str(rede_taxa_de_aprendizado)+"/modelo "+str(i+1)+"/config.txt"
		configFile = open(path, 'w')
		configFile.write("Execucao em "+time.strftime("%d/%m/%Y")+" "+time.strftime("%H:%M")+"\n")
		configFile.write("\n")
		configFile.write("extrator: HOG\n")
		configFile.write("extrator_orientacoes: 9\n")
		configFile.write("extrator_pixels_por_celula: 16\n")
		configFile.write("extrador_celulas_por_bloco: 1\n")
		configFile.write("\n")
		configFile.write("k: "+str(k)+"\n")
		configFile.write("rede_taxa_de_aprendizado: "+str(rede_taxa_de_aprendizado)+"\n")
		configFile.write("rede_camada_de_entrada_neuronios: "+str(rede_camada_de_entrada_neuronios)+"\n")
		#configFile.write("rede_camada_de_entrada_funcao_de_ativacao: sigmoide (logistica)")
		configFile.write("rede_camada_escondida_neuronios: "+str(rede_camada_escondida_neuronios)+"\n")
		configFile.write("rede_camada_escondida_funcao_de_ativacao: sigmoide (logistica)\n")
		configFile.write("rede_camada_de_saida_neuronios: "+str(rede_camada_de_saida_neuronios)+"\n")
		configFile.write("rede_camada_de_saida_funcao_de_ativacao: sigmoide (logistica)\n")
		configFile.write("rede_inicializacao_pesos: aleatoria\n")
		configFile.write("rede_min_epocas: "+str(rede_min_epocas)+"\n")
		configFile.write("rede_max_epocas: "+str(rede_max_epocas)+"\n")
		configFile.write("rede_parada_antecipada: nenhuma ou numero maximo de epocas")
		configFile.close()
		treinamento(modelos[i], imagens=indices[i*2], minEpoch=rede_min_epocas, maxEpoch=rede_max_epocas, validacao=indices[(i*2)+1], model_num=(i+1))

def cria_diretorio(model_num, filtro_de_imagens, mlp):	
	try:
		path = os.getcwd()+"/Hog/entrada = "+str(len(mlp.neurons[0]))+", escondida = "+str(len(mlp.neurons[1]))+", saida = "+str(len(mlp.neurons[2]))+", alpha = "+str(mlp.learningRate)+"/modelo "+str(model_num)+"/"
		truePath = os.path.dirname(path)
		os.makedirs(truePath)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise
	
def treinamento(mlp, imagens, minEpoch, maxEpoch, validacao, model_num):
	
	#print("Nós na entrada: "+str(len(mlp.neurons[0]))+"/neurônios na camada escondida: "+str(len(mlp.neurons[1]))+"/neurônios na camada de saída: "+str(len(mlp.neurons[2])))
	#print(str(len(imagens))+" imagens de treinamento")
	path = os.getcwd()+"/Hog/entrada = "+str(len(mlp.neurons[0]))+", escondida = "+str(len(mlp.neurons[1]))+", saida = "+str(len(mlp.neurons[2]))+", alpha = "+str(mlp.learningRate)+"/modelo "+str(model_num)+"/error.txt"

	errorFile = open(path, 'w')
	errorFile.write("Execucao em "+time.strftime("%d/%m/%Y")+" "+time.strftime("%H:%M")+"\n")
	errorFile.write("\n")
	errorFile.write("0;")
	aux = testaRede(mlp, imagens, epoca=0, tipo = 0)
	errorFile.write(str(aux)+";")
	testaRede(mlp, validacao, epoca=0, tipo = 1)	
	errorFile.write(str(aux)+"\n")
	for i in range(1, maxEpoch+1): # Loop de epoca
		#print(imagens)
		numResposta = 0
		startTime = time.time()
		for j in range(0, len(imagens)): # Loop de enumeracao da imagem
			if imagens[j]%3 == 0: charAtual = 'a'
			elif imagens[j]%3 == 1: charAtual = '3'
			else: charAtual = '8'
			# Determina a imagem que sera lida, aplica o descritor HoG na imagem e pega o vetor de features
			vetorDeFeatures = pegaVetorDeFeatures(imagens[j])		
			# Insere dados na rede neural
			insereDadosNaCamadaDeEntrada(mlp, vetorDeFeatures)	
			# Feedforward
			# Passa dados da camada entrada para camada escondida e guarda o valor evaluado pela funcao logistica
			passaDadosParaCamadaEscondida(mlp)
			# Passa sinais da camada escondida para camada de saida e guarda o valor evaluado pela funcao logistica
			passaDadosParaCamadaDeSaida(mlp)
			# Backpropagation
			# Determina os termos de informacao do erro na camada de saida
			termosDeInformacaoDeErroK = computaTermosDeInformacaoDeErroK(mlp, charAtual)
			# Determina os termos de informacao do erro na camada escondida	
			termosDeInformacaoDeErroJ = computaTermosDeInformacaoDeErroJ(mlp, termosDeInformacaoDeErroK)
			# Corrige os pesos e biases na camada de saida
			correcaoDosPesosEBiasesNaCamadaDeSaida(mlp, termosDeInformacaoDeErroK)
			# Corrige os pesos e biases na camada escondida
			correcaoDosPesosEBiasesNaCamadaEscondida(mlp, termosDeInformacaoDeErroJ)
		# Fim de uma epoca
		elapsedTime = time.time() - startTime
		print("Epoca "+str(i)+" demorou "+str(elapsedTime)+" segundos")
		print()
		errorFile.write(str(i)+";")
		aux = testaRede(mlp, imagens, epoca=i, tipo = 0)
		errorFile.write(str(aux)+";")
		aux = testaRede(mlp, validacao, epoca=i, tipo = 1)	
		errorFile.write(str(aux)+"\n")
		# Aleatoriza a ordem das imagens a cada época
		random.shuffle(imagens)
	
	errorFile.close()
	
	path = os.getcwd()+"/Hog/entrada = "+str(len(mlp.neurons[0]))+", escondida = "+str(len(mlp.neurons[1]))+", saida = "+str(len(mlp.neurons[2]))+", alpha = "+str(mlp.learningRate)+"/modelo "+str(model_num)+"/model.dat"
	pickle_out = open(path, "wb")
	pickle.dump(mlp, pickle_out)
	pickle_out.close()


def testaRede(mlp, imagens, epoca, tipo):
	respostasZ = 0
	respostasS = 0
	respostasX = 0
	erro = 0.0
	start_time = time.time()
	for j in range(0, len(imagens)): 
		aux = 0
		resposta, aux = tentaDefinirLetra(mlp, imagens[j])
		erro += aux
		if imagens[j]%3 == 0: charAtual = 'a'
		elif imagens[j]%3 == 1: charAtual = '3'
		else: charAtual = '8'
		if(resposta == "Z" and charAtual == 'a'): respostasZ = respostasZ + 1
		if(resposta == "S" and charAtual == '3'): respostasS = respostasS + 1
		if(resposta == "X" and charAtual == '8'): respostasX = respostasX + 1	
		#resetaValoresDosNeuronios(mlp)
	erro = erro/len(imagens)
	erro = erro/len(mlp.neurons[2])
	if(tipo == 0):
		print("Treinamento")
	elif(tipo == 1):
		print("Validação")
	print("Época "+str(epoca)+": Letras Z: "+str(respostasZ)+"/"+str(math.floor(len(imagens)/3))+" Letras S: "+str(respostasS)+"/"+str(math.floor(len(imagens)/3))+" Letras X: "+str(respostasX)+"/"+str(math.floor(len(imagens)/3))+" Total: "+str(respostasZ + respostasS + respostasX)+"/"+str(len(imagens)))	
	if(tipo == 0):
		print("Época "+str(epoca)+" erro médio de treinamento por imagem de cada neurônio: "+str(erro))
	elif(tipo == 1): 
		print("Época "+str(epoca)+" erro médio de validação por imagem de cada neurônio: "+str(erro))
	elapsed_time = time.time() - start_time
	if(tipo == 0):
		print("Para pegar o erro de treinamento demorou: "+str(elapsed_time)+" segundos")
	elif(tipo == 1): 
		print("Para pegar o erro de validacao demorou: "+str(elapsed_time)+" segundos")
	print()
	return erro		

# Função "não usada"
def resetaValoresDosNeuronios(mlp):
	for i in range(0, len(mlp.neurons)):
		for j in range(0, len(mlp.neurons[i])):
			mlp.neurons[i][j].value = 0 

def tentaDefinirLetra(mlp, index):
	if index%3 == 0: charDaImagem = 'a'
	elif index%3 == 1: charDaImagem = '3'
	else: charDaImagem = '8'
	# Determina a imagem que sera lida, aplica o descritor HoG na imagem e pega o vetor de features
	vetorDeFeatures = pegaVetorDeFeatures(index)				
	# Insere dados na rede neural
	insereDadosNaCamadaDeEntrada(mlp, vetorDeFeatures)
	# Feedforward
	# Passa dados da camada entrada para camada escondida e guarda o valor evaluado pela funcao logistica
	passaDadosParaCamadaEscondida(mlp)
	# Passa sinais da camada escondida para camada de saida e guarda o valor evaluado pela funcao logistica
	passaDadosParaCamadaDeSaida(mlp)
	# Calcula o erroMedio
	erroTotalDaImagem = 0
	erro = 0
	expectedIndex = 0
	if(charDaImagem == 'a'): expectedIndex = 0
	elif(charDaImagem == '3'): expectedIndex = 1
	else: expectedIndex = 2
	expected = 0
	for i in range(0, len(mlp.neurons[2])):
		if(i == expectedIndex): expected = 1
		else: expected = 0
		erro = expected - mlp.neurons[2][i].value
		erro = erro * erro
		erro = erro/2
		erroTotalDaImagem += erro
	#erroTotalDaImagem = math.sqrt(erroTotalDaImagem)
	#erroTotalDaImagem = erroTotalDaImagem/3
	#Calcula letra certa
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

	return resposta, erroTotalDaImagem

def pegaVetorDeFeatures(numero_imagem):
	if(numero_imagem%3 == 0): char_imagem = 'a'
	elif(numero_imagem%3 == 1): char_imagem = '3'
	else: char_imagem = '8'
	numero_string = str(math.floor(numero_imagem/3))
	while(len(numero_string)<4):
		numero_string = "0" + numero_string
	pickle_in = open("imagens_processadas/dataset1/treinamento/train_5"+char_imagem+"_0"+numero_string+".txt", "rb")
	vetor_de_features = pickle.load(pickle_in)
	pickle_in.close()
	return vetor_de_features

#def pegaVetorDeFeatures(numeroDaImagem):
	#if(numeroDaImagem%3 == 0): charDaImagem = 'a'
	#elif(numeroDaImagem%3 == 1): charDaImagem = '3'
	#else: charDaImagem = '8'
	# Corrige a numeracao da imagem para ficar no formato de 5 digitos (ex.: 999 para 00999)
	#trueFileNum = str(math.floor(numeroDaImagem/3))
	#while(len(trueFileNum)<4):
	#	trueFileNum = "0" + trueFileNum
	#Define o endereco e o nome da imagem que sera aberta
	#fileName = "dataset1/treinamento/train_5"+charDaImagem+"_0"+trueFileNum+".png"
	#print(fileName)
	#Abre a imagem
	#A = imread(fileName)
	#plt.figure()
	#imshow(A)
	#plt.show()
	#a1 = A[:,:,0]
	#Faz a descricao HoG da imagem
	#Retornos
	#x: vetor de features
	#y: imagem que pode ser plotada e visualizada
	#x, y = hog(a1, orientations = 9, pixels_per_cell = (16, 16), cells_per_block = (1, 1), visualise =  True)
	#return x

def insereDadosNaCamadaDeEntrada(mlp, vetorDeFeatures):
	#Coloca os dados na camada de entrada da rede neural
		for i in range(0, len(vetorDeFeatures)):
			mlp.neurons[0][i].value = vetorDeFeatures[i]

def passaDadosParaCamadaEscondida(mlp):
	#Passa os dados da camada de entrada para camada escondida
	for i in range(0, len(mlp.neurons[1])):
		totalSum = 0
		for j in range(0, len(mlp.neurons[0])):
			totalSum += mlp.neurons[0][j].weights[i] * mlp.neurons[0][j].value
		totalSum += mlp.neurons[1][i].bias
		#Evalua o sinal no neuronio na camada escondida usando a funcao logistica e guarda o resultado para passar para camada de saida
		result = evaluate2(totalSum)
		mlp.neurons[1][i].value = result	
	

def passaDadosParaCamadaDeSaida(mlp):
	#Passa os dados da camada escondida para camada de saida
	for i in range(0, len(mlp.neurons[2])):
		totalSum = 0
		for j in range(0, len(mlp.neurons[1])):
			totalSum += mlp.neurons[1][j].weights[i] * mlp.neurons[1][j].value
		totalSum += mlp.neurons[2][i].bias
		#Evalua o sinal no neuronio de saida usando a funcao logistica e guarda o resultado para calcular os erros
		result = evaluate2(totalSum)
		mlp.neurons[2][i].value = result


# Determina os termos de informacao do erro na camada escondida		
def computaTermosDeInformacaoDeErroJ(mlp, listaDeErros):

	listaDeErrosJ = []
	for i in range(0, len(mlp.neurons[1])):		
		totalSum = 0
		for j in range(0, len(mlp.neurons[2])):
			totalSum += mlp.neurons[1][i].weights[j] * listaDeErros[j]
		erro = mlp.neurons[1][i].value * (1 - mlp.neurons[1][i].value)
		erro = totalSum * erro
		listaDeErrosJ.append(erro)

	return listaDeErrosJ

# Determina os termos de informacao do erro na camada de saida
def computaTermosDeInformacaoDeErroK(mlp, charDaImagem):

	listaDeErrosK = []
	erro = 0
	expectedIndex = 0
	if(charDaImagem == 'a'): expectedIndex = 0
	elif(charDaImagem == '3'): expectedIndex = 1
	else: expectedIndex = 2
	expected = 0
	for i in range(0, len(mlp.neurons[2])):
		if(i == expectedIndex): expected = 1
		else: expected = 0
		# erro = (valor esperado no neuronio de saida - valor obtido no neuronio de saida) * derivada da funcao sigmoid
		erro = (expected - mlp.neurons[2][i].value) * (mlp.neurons[2][i].value * (1 - mlp.neurons[2][i].value)) 
		listaDeErrosK.append(erro)

	return listaDeErrosK
	
def correcaoDosPesosEBiasesNaCamadaEscondida(mlp, termosDeInformacaoDeErroJ):
	#Corrige os pesos entre a camada de entrada e a camada escondida
		for i in range(0, len(mlp.neurons[0])):
			for j in range(0, len(mlp.neurons[1])):
				mlp.neurons[0][i].weights[j] += mlp.learningRate * mlp.neurons[0][i].value * termosDeInformacaoDeErroJ[j]
			
	#Corrige os bias entre a camada de entrada e a camada escondida
		for i in range(0, len(mlp.neurons[1])):
			mlp.neurons[1][i].bias += mlp.learningRate * termosDeInformacaoDeErroJ[i]		

def correcaoDosPesosEBiasesNaCamadaDeSaida(mlp, termosDeInformacaoDeErroK):
	#Corrige os pesos entre a camada de saida e a camada escondida, isto e, soma os pesos com seus respectivos gradientes
	for i in range(0, len(mlp.neurons[1])):
		for j in range(0, len(mlp.neurons[2])):
			mlp.neurons[1][i].weights[j] += mlp.learningRate * mlp.neurons[1][i].value * termosDeInformacaoDeErroK[j]
			
	#Corrige os biases entre a camada de saida e a camada escondida, isto e, soma os biases com seus respectivos gradientes
	for i in range(0, len(mlp.neurons[2])):
		mlp.neurons[2][i].bias += mlp.learningRate * termosDeInformacaoDeErroK[i]	

# Divide o conjunto do indices das imagens conforme o k-fold cross validation

# Argumentos:
# k: numeros de folds para realizar o k-fold cross validation
# numImagem: enumeracao maxima das imagens (1000, pois as imagens vao de 00000 ate 00999 na pasta "treinamento")
# boolRandom: aleatoriza cada fold se for True

# Retorno:
# Cria uma lista de listas de numeros inteiros chamada "retorno"
# A lista "retorno" tera 2*k listas de numeros inteiros
# Esses numeros inteiros sao os numeros das imagens
# Na pasta treinamento cada imagem tem numero de 00000 ate 00999, porém há 1 imagem de cada letra
# Logo temos 3000 imagens na pasta treinamento
# Os indices pares (0, 2, ...) da lista retorno sao as k-1 partes do treinamento
# Os indices impares (1, 3, ...) da lista retorno sao as partes unicas usadas para validacao

def kFoldCrossValidation(k, numImagem, boolRandom):
	# Faz uma lista com inteiros, por exemplo: [0, 1, 2, ..., 999], se numImagem = 1000
	indices = []
	for i in range(0, numImagem): 
		indices.append(i)
	# Faz e preenche a lista "retorno" e suas listas de numeros inteiros
	retorno = []
	for i in range(0, k):
		aux = []
		if(i == 0):
			for j in range(math.floor(numImagem*(i+1)/k), numImagem):
				aux.append(indices[j])
		elif(i == k - 1):
			for j in range(0, math.floor(numImagem*(k-1)/k)):
				aux.append(indices[j])
		else:
			for j in range(0, math.floor(numImagem*i/k)):
				aux.append(indices[j])
			for j in range(math.floor((numImagem*(i+1))/k), numImagem):
				aux.append(indices[j])
		retorno.append(aux)
		aux = []
		for j in range(math.floor(numImagem*i/k), math.floor(numImagem*(i+1)/k)):
			aux.append(indices[j])
		retorno.append(aux)
	# Randomiza a ordem das imagens no fold se boolRandom=True
	if(boolRandom):
		for i in range(0, len(retorno)):
			random.shuffle(retorno[i])	

	return retorno

if __name__ == "__main__": 
	main()
