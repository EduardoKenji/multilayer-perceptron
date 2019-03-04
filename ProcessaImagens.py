import numpy as np
import os
import errno
import pickle
import math
from skimage.io import imread, imshow
from skimage.feature import hog

def main():
	cria_diretorio(os.getcwd()+"/imagens_processadas/dataset1/treinamento/")
	for i in range(0, 3000):
		if(i%3 == 0): char_imagem = 'a'
		elif(i%3 == 1): char_imagem = '3'
		else: char_imagem = '8'
		numero_string = str(math.floor(i/3))
		while(len(numero_string)<4):
			numero_string = "0" + numero_string
		processa_imagem_HOG("dataset1/treinamento/train_5"+char_imagem+"_0"+numero_string+".png", "imagens_processadas/dataset1/treinamento/train_5"+char_imagem+"_0"+numero_string+".txt", 16, 1, 9)
	
def cria_diretorio(caminho_diretorio):	
	try:
		truePath = os.path.dirname(caminho_diretorio)
		os.makedirs(truePath)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise

def processa_imagem_HOG(caminho_da_imagem, caminho_de_saida, pixels_celula, celulas_bloco, orientacoes):
	A = imread(caminho_da_imagem)
	a1 = A[:,:,0]
	x = hog(a1, orientations = orientacoes, pixels_per_cell = (pixels_celula, pixels_celula), cells_per_block = (celulas_bloco, celulas_bloco), visualise = False)
	pickle_out = open(caminho_de_saida, "wb")
	pickle.dump(x, pickle_out)
	pickle_out.close()
	print(caminho_da_imagem)
	print(caminho_de_saida)

if __name__ == "__main__": 
	main()
