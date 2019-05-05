"""
@author: María José Belda Beneyto
@author: Miguel Pascual Domínguez
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
        clasificador: Calcula las clases estimadas de los puntos que
            se le pasan
    """
    
    """
    Inicialización de los atributos de la clase por defecto
    """
    def __init__(self):
        self.W = []
    
    """
    Calcula la matriz W asociada al clasificador y la guarda en el atributo
    de la clase
    
    Entrada
        X: Puntos
        T: Clases asociadas a los puntos de X
    """
    def calculaW(self, X, T):
        Xg = np.vstack((np.ones((1, X.shape[1])), X))
        self.W = np.linalg.solve(Xg.dot(Xg.T),Xg.dot(T.T))
    

    """
    Clasifica los puntos dados, es decir, calcula las etiquetas que les
    corresponden usando la matriz W de la clase
    
    Entrada
        puntos: puntos para clasificar
    Salida
        Devuelve las clases predecidas de los puntos
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

Entrada
    K: número de clases
    distClases: distancia mínima entre las clases
    dispMedia: dispersión media de cada clase
    minNxClase: mínimo número de puntos de cada clase
    maxNxClase: máximo número de puntos de cada clase

Salida
    X: matriz con los puntos creados
    T: matriz con las etiquetas asociadas a los puntos de X
"""    
def creaDatos(K, distClases, dispMedia, minNxClase, maxNxClase):
       
    """Generamos las mu distanciadas para que las clases esten separadas"""
    mus = [np.random.randn(2)*distClases]
    for k in range(1, K):
        auxMu = np.random.randn(2)*distClases
        while min([np.linalg.norm(mu - auxMu) for mu in mus]) < 2*distClases:
            auxMu = np.random.randn(2)*distClases
        mus.append(auxMu)
      
    """Generamos el numero de datos que tendrá cada clase"""    
    Ns = []
    for k in range(0, K):
        Ns.append(np.random.randint(minNxClase, maxNxClase))
    N = sum(Ns)
    
    """Rellenamos las etiquetas para cada punto"""
    T = np.zeros((K,N))
    cont = 0
    for k in range(0, K):
        T[k, cont:cont + Ns[k]] = np.ones(Ns[k])
        cont += Ns[k]
    
    """Generamos los puntos"""
    X = np.zeros((2, N))
    cont = 0
    for k in range(0, K):
        X[:, cont:cont + Ns[k]] = np.random.randn(2, Ns[k])*dispMedia + mus[k][:, np.newaxis]
        cont += Ns[k]
    
    
    return X, T
    
if __name__ == '__main__':
    mcc = MinimosCuadrados()
    """Número de clases"""
    K = 3 
    """Distancia mínima entre las clases"""
    distClases = 10 
    """Dispersión de cada clase"""
    dispMedia = 0.8
    """Cotas del número de puntos de cada clase"""
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