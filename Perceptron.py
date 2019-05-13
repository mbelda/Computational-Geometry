"""
@author: María José Belda Beneyto
@author: Miguel Pascual Domínguez
"""

import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """
    Atributos:
        W: matriz del clasificador
        
    Metodos:
        calculaW: calcula la matriz del clasificador dados unos puntos y sus
            correspondientes clases
        clasificador: Usando W calcula las clases estimadas de los puntos que
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
        F, N = X.shape
        wr = np.zeros(F+1)
        wrIgualwr1 = False
        auxX = np.zeros((F+1,N))
        auxX[:2,:]= X
        auxX[2,:] = np.ones(N)
        print(auxX)
        while not(wrIgualwr1):
            for i in range(N):
                if (wr.T).dot(auxX[:,i])*T[i] <= 0 :
                    #Mal clasificado
                    wr1 = wr + (auxX[:,i]*T[i]).T                        
            if np.equal(wr1, wr).all():
                wrIgualwr1 = True
            else:
                wr = np.copy(wr1)
        print(wr.shape)
        self.W = wr

    """
    Clasifica los puntos dados, es decir, calcula las etiquetas que les
    corresponden usando la matriz W de la clase
    
    Entrada
        X: puntos para clasificar
    Salida
        Devuelve las clases predecidas de los puntos
    """
    def clasificador(self, X):
        F, N = X.shape
        auxX = np.zeros((F+1,N))
        auxX[:2,:]= X
        auxX[2,:] = np.ones(N)
        T = np.ones((1,N))
        for i in range(N):
            if (self.W.T).dot(auxX[:,i]) < 0:
                T[0, i] = (-1) * np.ones(1)
        return T

"""
Crea datos aleatoriamente pero controlamos que los puntos de la misma clase
tengan cierta relación para que el clasificador funcione razonablemente bien

Entrada
    distClases: distancia mínima entre las clases
    dispMedia: dispersión media de cada clase
    minNxClase: mínimo número de puntos de cada clase
    maxNxClase: máximo número de puntos de cada clase

Salida
    X: matriz con los puntos creados
    T: matriz con las etiquetas asociadas a los puntos en X
"""    
def creaDatos(distClases, dispMedia, minNxClase, maxNxClase):
       
    """Generamos las mu distanciadas para que las clases esten separadas"""
    mus = [np.random.randn(2)*distClases]
    for k in range(1, 2):
        auxMu = np.random.randn(2)*distClases
        while min([np.linalg.norm(mu - auxMu) for mu in mus]) < 2*distClases:
            auxMu = np.random.randn(2)*distClases
        mus.append(auxMu)
      
    """Generamos el numero de datos que tendrá cada clase"""    
    Ns = []
    for k in range(0, 2):
        Ns.append(np.random.randint(minNxClase, maxNxClase))
    N = sum(Ns)
    
    """Rellenamos las etiquetas para cada punto"""
    T = np.hstack((np.ones(Ns[0]), (-1)*np.ones(Ns[1])))
    
    """Generamos los puntos"""
    X = np.zeros((2, N))
    cont = 0
    for k in range(0, 2):
        X[:, cont:cont + Ns[k]] = np.random.randn(2, Ns[k])*dispMedia + mus[k][:, np.newaxis]
        cont += Ns[k]
    
    
    return X, T
    
if __name__ == '__main__':
    per = Perceptron()
    
    """Distancia mínima entre las clases"""
    distClases = 10 
    """Dispersión de cada clase"""
    dispMedia = 0.8
    """Cotas del número de puntos de cada clase"""
    minNxClase = 50
    maxNxClase = 100
    
    """Entrenamiento"""
    X, T = creaDatos(distClases, dispMedia, minNxClase, maxNxClase)
    per.calculaW(X, T)
    
    """Caso de prueba"""
    Xs = np.linspace(-15,15,200)
    Ys = np.linspace(-15,15,200)
    XX, YY = np.meshgrid(Xs, Ys)
    x = XX.flatten()
    y = YY.flatten()
    puntos = np.vstack((x, y))

    ZZ = per.clasificador(puntos).reshape((200,200))
    plt.plot(X[0, :], X[1, :], 'o')
    #plt.contour(XX, YY, ZZ)
    a = np.min(X[0])
    b = np.max(X[0])
    plt.plot([a, b], [(-per.W[0]*a - per.W[2])/per.W[1], (-per.W[0]*b - per.W[2])/per.W[1]])