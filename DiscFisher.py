# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 22:34:10 2019

@author: Majo
"""

import numpy as np
import matplotlib.pyplot as plt

class DiscFisher:
    """
    Atributos:
        w: vector de la proyección
        
    Metodos:
        calculaW: calcula el vector del clasificador dados unos puntos y sus
            correspondientes clases
        proyecta: Usando W calcula la proyeccion de los puntos que se le pasan
    """
    
    """
    Inicialización de los atributos de la clase por defecto
    """
    def __init__(self):
        self.w = []
    
    """
    Calcula el vector w asociado a la proyección mediante los argumentos de
    entrada y lo guarda en el atributo de la clase
    
    Entrada
        X: Puntos
        T: Clases asociadas a los puntos en X
    """
    def calculaW(self, X, medias, Ns):
        """Calculamos Sw """
        #Sw = Sw1 + Sw2
        #Swk = suma n en Ck de (xn - mk)(xn - mk)T
        #Sabemos que los Ns[0] primeros datos de X son los de la clase 1
        F, C = X.shape
        Sw1 = np.zeros((2,2))
        for i in range(0, Ns[0]):
            Sw1 += (X[:,i] - medias[0])*((X[:,i] - medias[0]).T)
        Sw2 = np.zeros((2,2))
        for i in range(Ns[0], Ns[0] + Ns[1]):
            Sw2 += (X[:, i] - medias[1])*((X[:, i] - medias[1]).T)
        Sw = Sw1 +Sw2    
        
        """Calculamos w"""
        self.w, residuals, rank, s = np.linalg.lstsq(Sw, medias[1] - medias[0])
        


    def proyecta(self, X, Ns):
        Xp = []
        N = sum(Ns)
        for i in range(0, N):
            aux = (self.w.T).dot(X[:, i])
            Xp.append(aux)
        
        return Xp
    
    def calculaC(self, Ns, v, m):
        """Queremos calcular als raíces del polinomio F(c)"""
        p0 = Ns[0]/sum(Ns)
        p1 = Ns[1]/sum(Ns)
        coefs = [(v[0]**2 * m[1]**2 - v[1]**2 * m[0]**2)/(2 * v[0]**2 * v[1]**2)
                    + np.log(p0/v[0]) - np.log(p1/v[1]),
                (m[0] * v[1]**2 - m[1] * v[0]**2)/(v[0]**2 * v[1]**2),
                (v[0]**2 - v[1]**2)/(2 * v[0]**2 * v[1]**2)]
        raices = np.roots(coefs)
        print(raices)
        #Falta que esto funcione bien y mirar que raiz es el minimo
        #(hay a lo sumo 2 raices)
        return raices[0]
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
        cols = puntos.shape[1]
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
def creaDatos(distClases, minNxClase, maxNxClase):
    
    """Generamos la varianza de cada clase"""
    varianzas = []
    for k in range(0,2):
        varianzas.append(np.random.random()) 
    
    """Generamos las mu distanciadas para que las clases esten separadas"""
    mus = [np.random.randn(2)*distClases] 
    auxMu = np.random.randn(2)*distClases
    while np.linalg.norm(mus[0] - auxMu) < 2*distClases:
        auxMu = np.random.randn(2)*distClases
    mus.append(auxMu)
      
    """Generamos el numero de datos que tendra cada clase"""    
    Ns = []
    for k in range(0, 2):
        Ns.append(np.random.randint(minNxClase, maxNxClase))       
    N = sum(Ns)
    
    """Rellenamos las etiquetas para cada punto"""
    T = np.zeros((2,N))
    cont = 0
    for k in range(0, 2):
        T[k, cont:cont + Ns[k]] = np.ones(Ns[k])
        cont += Ns[k]
    
    """Generamos los puntos"""
    X = np.zeros((2, N))
    medias = []
    cont = 0
    for k in range(0, 2):
        X[:, cont:cont + Ns[k]] = np.random.randn(2, Ns[k])*varianzas[k] + mus[k][:, np.newaxis]
        aux = 0
        for i in range(cont, cont + Ns[k]):
            aux += X[:, i]
        medias.append(aux/Ns[k])
        cont += Ns[k]
        
    return X, T, Ns, varianzas, medias
    
if __name__ == '__main__':
    df = DiscFisher()
    distClases = 10 
    minNxClase = 50
    maxNxClase = 100
    
    X, T, Ns, varianzas, medias = creaDatos(distClases, minNxClase, maxNxClase)
    plt.plot(X[0, :], X[1, :], 'o')
    df.calculaW(X, medias, Ns)
    Xp = df.proyecta(X, Ns)
    plt.plot(Xp, '-o') 
    c = df.calculaC(Ns, varianzas, medias)
    
#    Xs = np.linspace(-15,15,200)
#    Ys = np.linspace(-15,15,200)
#    XX, YY = np.meshgrid(Xs, Ys)
#    x = XX.flatten()
#    y = YY.flatten()
#    puntos = np.vstack((x, y))
#      
#    ZZ = df.clasificador(puntos).reshape((200,200))
#    plt.plot(X[0, :], X[1, :], 'o')
#    plt.contour(XX, YY, ZZ)