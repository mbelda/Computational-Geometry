# -*- coding: utf-8 -*-
"""

@author: María José Belda Beneyto
@author: Miguel Pascual Domínguez
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
        medias: medias de cada clase
        T: Clases asociadas a los puntos en X
    """
    def calculaW(self, X, medias, Ns):
        """Calculamos Sw """
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
        
    """
    Proyecta a R usando w los puntos que se le pasan
    
    Entrada
        X: puntos a proyectar
    
    Salida
        Xp: resultado de los puntos proyectados
    """ 
    def proyecta(self, X):   
        F, N = X.shape
        Xp = np.zeros((1,N))
        for i in range(0, N):
            aux = (self.w.T).dot(X[:, i])
            Xp[:, i] = aux
        
        return Xp
    
    """
    Calcula el punto c de R que marca la división entre las proyecciones
    de los puntos de las clases. Este punto es una raíz del polinomio que definimos
    
    Entrada
        Ns: Número de puntos de las clases
        v: varianzas de las proyecciones
        m: medias de las proyecciones
        
    Salida
        c  
    """
    def calculaC(self, Ns, v, m):
        p0 = Ns[0]/sum(Ns)
        p1 = Ns[1]/sum(Ns)
        coefs = [(v[0]**2 - v[1]**2)/(2 * v[0]**2 * v[1]**2),
                 (m[0] * v[1]**2 - m[1] * v[0]**2)/(v[0]**2 * v[1]**2),
                 (v[0]**2 * m[1]**2 - v[1]**2 * m[0]**2)/(2 * v[0]**2 * v[1]**2)
                    + np.log(p0/v[0]) - np.log(p1/v[1])]
        raices = np.roots(coefs)
        print(raices)
        coefs2 = [(v[0]**2-v[1]**2)/(v[0]**2 * v[1]**2), (m[0]*v[1]**2-m[1]*v[0]**2)/(v[0]**2 * v[1]**2)]
        
        if np.polyval(coefs2, raices[0]) < np.polyval(coefs2, raices[1]):
            return raices[0]
        else:    
            return raices[1]

"""
Crea datos aleatoriamente pero controlamos que los puntos de la misma clase
tengan cierta relación para que el clasificador funcione razonablemente bien

Entrada
    distClases: distancia minima entre las clases
    minNxClase: mínimo número de puntos por clase
    maxNxClase: máximo número de puntos por clase

Salida
    X: matriz con los puntos creados.
    T: matriz con las etiquetas asociadas a los puntos en X.
	Ns: lista con la cantidad de datos que hay de cada clase.
	Varianzas: lista con las varianzas de cada clase.
	Medias: lista con las medias por clases de los datos.
"""    
def creaDatos(distClases, minNxClase, maxNxClase):
    
    """Generamos la varianza de cada clase"""
    varianzas = np.zeros(2)
    for k in range(0,2):
        varianzas[k]= np.random.random()
    
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
    medias = np.zeros((2,2))
    cont = 0
    for k in range(0, 2):
        X[:, cont:cont + Ns[k]] = np.random.randn(2, Ns[k])*varianzas[k] + mus[k][:, np.newaxis]
        aux = 0
        for i in range(cont, cont + Ns[k]):
            aux += X[:, i]
        medias[k] = aux/Ns[k]
        cont += Ns[k]
        
    return X, T, Ns, varianzas, medias

"""
Cálcula la media de cada clase y las guarda en un vector.

Entrada
    Xp: matriz con los puntos
    Ns: vector de número de puntos de cada clase

Salida
    mp: vector con las medias de cada clase
""" 
def calculoMedias(Xp, Ns):
    mp = np.zeros(2)
    cont = 0
    for k in range(2):
        aux = 0
        for i in range(cont, cont + Ns[k]):
            aux += Xp[:, i]
        mp[k] = aux/Ns[k]
        cont += Ns[k]
    
    return mp

        
    
if __name__ == '__main__':
    df = DiscFisher()
    distClases = 10 
    minNxClase = 50
    maxNxClase = 100
    
    X, T, Ns, varianzas, medias = creaDatos(distClases, minNxClase, maxNxClase)
    plt.plot(X[0, :], X[1, :], 'o')
    df.calculaW(X, medias, Ns)
    
    Xp = df.proyecta(X)
    mp = calculoMedias(Xp,Ns)
    c = df.calculaC(Ns, varianzas, mp)    
    
    t = np.array([-20,10,20])
    plt.plot(t,(c - df.w[0]*t)/df.w[1])