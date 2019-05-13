# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:23:29 2019

@author: Miguel
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


class PCA:
    
    def calculaSVD(self,datos):
        """
        Calcula la descomposición en valores singulares 
        de una matriz de datos usando la función de numpy
        de svd.
        Entrada:
            datos: la matriz de datos
        Salida:
            U: matriz U de la descomposición SVD
            D2: vector compuesto de los valores singulares.
            Vt: matriz Vt de la descomposición SVD
        """
        U, D2, Vt = np.linalg.svd(datos, full_matrices=False)        
        return U, D2, Vt

    def calculadprima(self,D2,D,umbral):
        conseguido = False
        sumVarSing = np.sum(D2)
        i=1
        while i<D and not(conseguido):
            conseguido = np.sum(D2[i:])/sumVarSing < umbral
            i = i + 1
        return i



def mostrarDatos(U,Dprima,datos):
    figura = plt.figure()
    figura.subplots_adjust(hspace=0.4, wspace=0.4)


if __name__ == '__main__':
    pca = PCA()
    mnist = loadmat(r'C:\Users\Miguel\mnist-original.mat')
    datos = mnist['data']
    #Dimension 
    D = 784
    #Número de datos
    N = 70000
    #umbral
    umbral = 0.15
    #Centremos los datos en la media
    media = np.mean(datos, axis = 1)
    Xcentrados = datos - media[:, np.newaxis]
    
    #Calculemos la descomposición SVD de los datos centrados en la media 
    U, D2, Vt = pca.calculaSVD(Xcentrados)
    #Escogamos la dimensión D' mas pequeña que siga mateniendo una estructura razonable en función del umbral epsilon.
    dprima = pca.calculadprima(D2,D,umbral)
    #Comprimamos ahora todos los datos a la dimensión D' que es la óptima
    
    
    
