# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 10:36:01 2019

@author: María José Belda Beneyto y Miguel Pascual Domínguez
"""

import numpy as np
import matplotlib.pyplot as plt

class MinimosCuadrados:
    """
    Atributos:
        W: matriz del clasificador
        
    Metodos:
        calculaW: calcula la matriz del clasificador dados unos puntos y sus
            correspondientes clases
        casificador: Usando W calcula las clases estimadas de los puntos que
            se le pasan
    """
    
    """
    Inicialización de los atributos de la clase por defecto
    """
    def __init__(self):
        self.W = []
    
    """
    Calcula la matriz W asociada al clasificador mediante los argumentos de
    entrada y la guarda en el atributo de la clase
    
    Entrada
        X: Puntos
        T: Clases asociadas a los puntos en X
    """
    def calculaW(self, X, T):
        F, N = X.shape
        wr = np.zeros(F)
        wrIgualwr1 = False

        while not(wrIgualwr1):
            for i in range(N):
                if (wr.T).dot(X[:,i])*T[i] <= 0 :
                    #Mal clasificado
                    wr1 = wr + X[:,i]*T[i]                        
            if np.equal(wr1, wr).all():
                wrIgualwr1 = True
            else:
                wr = np.copy(wr1)
        
        self.W = wr

    """
    Clasifica los puntos dados, es decir, calcula las etiquetas que les
    corresponden usando la matriz W de la clase
    
    Entrada
        puntos
    """
    def clasificador(self, X):
        F, N = X.shape
        T = np.ones((1,N))
        for i in range(N):
            if (self.W.T).dot(X[:,i]) < 0:
                T[0, i] = (-1) * np.ones(1)
        return T

"""
Crea datos aleatoriamente pero controlamos que los puntos de la misma clase
tengan cierta relación para que el clasificador funcione razonablemente bien


Salida
    X: matriz con los puntos creados
    T: matriz con las etiquetas asociadas a los puntos en X
"""    
def creaDatos(K, distClases, dispMedia, minNxClase, maxNxClase):
       
    """Generamos las mu distanciadas para que las clases esten separadas"""
    mus = [np.random.randn(2)*distClases]
    for k in range(1, K):
        auxMu = np.random.randn(2)*distClases
        while min([np.linalg.norm(mu - auxMu) for mu in mus]) < 2*distClases:
            auxMu = np.random.randn(2)*distClases
        mus.append(auxMu)
      
    """Generamos el numero de datos que tendra cada clase"""    
    Ns = []
    for k in range(0, K):
        Ns.append(np.random.randint(minNxClase, maxNxClase))
    N = sum(Ns)
    
    """Rellenamos las etiquetas para cada punto"""
    T = np.hstack((np.ones(Ns[0]), (-1)*np.ones(Ns[1])))
    
    """Generamos los puntos"""
    X = np.zeros((2, N))
    cont = 0
    for k in range(0, K):
        X[:, cont:cont + Ns[k]] = np.random.randn(2, Ns[k])*dispMedia + mus[k][:, np.newaxis]
        cont += Ns[k]
    
    
    return X, T
    
if __name__ == '__main__':
    mcc = MinimosCuadrados()
    K = 2 
    distClases = 10 
    dispMedia = 0.8
    minNxClase = 50
    maxNxClase = 100
    
    """Entrenamiento"""
    X, T = creaDatos(K, distClases, dispMedia, minNxClase, maxNxClase)
    mcc.calculaW(X, T)
    
    """Caso de prueba"""
    Xs = np.linspace(-15,15,200)
    Ys = np.linspace(-15,15,200)
    XX, YY = np.meshgrid(Xs, Ys)
    x = XX.flatten()
    y = YY.flatten()
    puntos = np.vstack((x, y))
      
    ZZ = mcc.clasificador(puntos).reshape((200,200))
    plt.plot(X[0, :], X[1, :], 'o')
    plt.contour(XX, YY, ZZ)