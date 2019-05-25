# -*- coding: utf-8 -*-
"""
@author: Miguel Pascual Domínguez
@author: María José Belda Benyto
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
        """
        Calcula la mínima dimensión a la que podemos comprimir los datos, 
        sin eliminar más información de la que nos permite el umbral.
        Entrada:
            D2: matriz D de la descomposición
            D: dimensión de los datos originales
            umbral: número que nos indica cuanta información que podemos quitar
        Salida:
            i: mínima dimensión a la que se pueden comprimir los datos
        """
        conseguido = False
        sumVarSing = np.sum(D2)
        i=1
        while i<D and not(conseguido):
            conseguido = np.sum(D2[i:])/sumVarSing < umbral
            i = i + 1
        return i
    
    def comprimirDatos(self,U,dprima,datos,media):
        """
        Calcula las coordenadas de los datos coomprimidos
        Entrada:
            U: matriz U de la descomposición SVD
            dprima: dimensión a la que queremos comprimir
            datos: datos cogidos de la base de datos mnist
            media: media de los datos originales
        Salida:
            datosCompri: datos comprimidos 
        """
        Xcentrados = datos - media[:,np.newaxis]
        
        UUt = U[:,:dprima].dot(U[:,:dprima].T) 
        
        datosCompri = media[:,np.newaxis] + UUt.dot(Xcentrados)
        return datosCompri


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
    datosCompri= pca.comprimirDatos(U,dprima,datos,media)
    #Mostremos parte de los datos Comprimido
    fig, axs = plt.subplots(2, 3, figsize=(20, 20))
    for k, ax in enumerate(axs[0, :]):
        ax.imshow(datosCompri[:, k].reshape(28, 28), cmap='gray')
        if k == 0: ax.set_ylabel('D\'=%d'.ljust(20)%dprima, fontsize=50, rotation=0)
    #Mostramos la "misma" parte los datos originales
    for k, ax in enumerate(axs[-1, :]):
        ax.imshow(datos[:, k].reshape(28, 28), cmap='gray')
        if k == 0: ax.set_ylabel('Original'.ljust(20), fontsize=50, rotation=0)