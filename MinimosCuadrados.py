# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 10:36:01 2019

@author: Maria Jose Belda Beneyto
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
        Xg = np.vstack((np.ones((1, X.shape[1])), X))
        self.W = np.linalg.solve(Xg.dot(Xg.T),Xg.dot(T.T))
    

    """
    Clasifica los puntos dados, es decir, calcula las etiquetas que les
    corresponden usando la matriz W de la clase
    
    Entrada
        puntos
    """
    def clasificador(self, puntos):
        if self.W is []:
            print("Entrena al clasificador")
            return
        filas, cols = puntos.shape
        Puntosg = np.vstack((np.ones(cols), puntos))
        T = (self.W.T).dot(Puntosg)
        return np.argmax(T, axis=0)

"""
Crea datos aleatoriamente pero controlamos que los puntos de la misma clase
tengan cierta relación para que el clasificador funcione razonablemente bien


Salida
    X: matriz con los puntos creados
    T: matriz con las etiquetas asociadas a los puntos en X
"""    
def creaDatos(K):
    distClases = 10
    ndatosxClase = 100
    dispMedia = 0.8
    
    mus = [np.random.randn(2)*distClases]
    for k in range(1, K):
        auxMu = np.random.randn(2)*distClases
        while min([np.linalg.norm(mu - auxMu) for mu in mus]) < 2*distClases:
            auxMu = np.random.randn(2)*distClases
        mus.append(auxMu)
    
    Ns = []
    for k in range(0, K):
        Ns.append(ndatosxClase)
    N = sum(Ns)
    
    X = np.zeros((2, N))
    cont = 0
    for k in range(0, K):
        X[:, cont:cont + Ns[k]] = np.random.randn(2, Ns[k])*dispMedia + mus[k][:, np.newaxis]
        cont += Ns[k]
    
    
    Unos = np.ones((1,100))
    Ceros = np.zeros((1,100))
    T1 = np.hstack((Unos,Ceros,Ceros))
    T2 = np.hstack((Ceros,Unos,Ceros))
    T3 = np.hstack((Ceros,Ceros,Unos))
    T = np.vstack((T1,T2,T3))
    return X, T
    
if __name__ == '__main__':
    mcc = MinimosCuadrados()
    K = 3
    X, T = creaDatos(K)
    mcc.calculaW(X, T)
    
    Xs = np.linspace(-15,15,200)
    Ys = np.linspace(-15,15,200)
    XX, YY = np.meshgrid(Xs, Ys)
    x = XX.flatten()
    y = YY.flatten()
    puntos = np.vstack((x, y))
      
    ZZ = mcc.clasificador(puntos).reshape((200,200))
    plt.plot(X[0, :], X[1, :], 'o')
    plt.contour(XX, YY, ZZ)