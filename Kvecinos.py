"""
@author: María José Belda Beneyto
@author: Miguel Pascual Domínguez
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

    
def estima(K, X, T, Ns, C, punto):       
    minDistVecinoClasek = np.full(C,sys.float_info.max)
    Ks = np.zeros(C)
    for i in range(K):
        k = np.argmax(T[:,i])
        Ks[k] += 1
        minDistVecinoClasek[k] = min(minDistVecinoClasek[k],
                                     np.linalg.norm(punto - X[:,i]))
    
    N = sum(Ns)
    for i in range(K, N):
        auxVecinos = np.copy(minDistVecinoClasek)
        for j in range(C):
            if Ks[j] == 0: 
                auxVecinos[j] = 0
                
        if(np.linalg.norm(punto - X[:,i]) < np.amax(auxVecinos)) :
            #He encontrado uno mas cerca
            kMejor = np.argmax(T[:,i])
            kViejo = np.argmax(auxVecinos)
            Ks[kViejo] -= 1
            Ks[kMejor] += 1
            minDistVecinoClasek[kMejor] = min(minDistVecinoClasek[kMejor],
                                              np.linalg.norm(punto - X[:,i]))
    etiqueta = np.zeros(C)
    etiqueta[np.argmax(Ks)] = 1
    return etiqueta        

def creaDatos(C, distClases, dispMedia, minNxClase, maxNxClase):
    """
    Crea datos aleatoriamente pero controlamos que los puntos de la misma clase
    tengan cierta relación para que el clasificador funcione razonablemente bien
    
    Entrada
        C: número de clases
        distClases: distancia mínima entre las clases
        dispMedia: dispersión media de cada clase
        minNxClase: mínimo número de puntos de cada clase
        maxNxClase: máximo número de puntos de cada clase
    
    Salida
        X: matriz con los puntos creados
        T: matriz con las etiquetas asociadas a los puntos de X
    """  
    
    """Generamos las mu distanciadas para que las clases esten separadas"""
    mus = [np.random.randn(2)*distClases]
    for k in range(1, C):
        auxMu = np.random.randn(2)*distClases
        while min([np.linalg.norm(mu - auxMu) for mu in mus]) < 2*distClases:
            auxMu = np.random.randn(2)*distClases
        mus.append(auxMu)
      
    """Generamos el numero de datos que tendrá cada clase"""    
    Ns = []
    for k in range(0, C):
        Ns.append(np.random.randint(minNxClase, maxNxClase))
    N = sum(Ns)
    
    """Rellenamos las etiquetas para cada punto"""
    T = np.zeros((C,N))
    cont = 0
    for k in range(0, C):
        T[k, cont:cont + Ns[k]] = np.ones(Ns[k])
        cont += Ns[k]
    
    """Generamos los puntos"""
    X = np.zeros((2, N))
    cont = 0
    for k in range(0, C):
        X[:, cont:cont + Ns[k]] = np.random.randn(2, Ns[k])*dispMedia + mus[k][:, np.newaxis]
        cont += Ns[k]
    
    
    return X, T, Ns, mus
    
if __name__ == '__main__':
    """Número de clases"""
    C = 3 
    """Distancia mínima entre las clases"""
    distClases = 10 
    """Dispersión de cada clase"""
    dispMedia = 0.8
    """Cotas del número de puntos de cada clase"""
    minNxClase = 50
    maxNxClase = 100
    """Número de vecinos"""
    K = 10
    
    """Entrenamiento"""
    X, T, Ns, mus = creaDatos(C, distClases, dispMedia, minNxClase, maxNxClase)
    
    """Dibujar con colores, solo sirve para C = 3"""
    plt.plot(X[0, 0:Ns[0]], X[1, 0:Ns[0]], 'bo')
    plt.plot(X[0, Ns[0]:Ns[0]+Ns[1]], X[1, Ns[0]:Ns[0]+Ns[1]], 'ro')
    plt.plot(X[0, Ns[0]+Ns[1]:], X[1, Ns[0]+Ns[1]:], 'go')
    for i in range(10): 
        """Caso de prueba"""
        punto = (2*distClases*np.random.randn(1))*np.random.random(2)
        etiquetaEstimadaX = estima(K, X, T, Ns, C, punto)
        claseEstimadaX = np.argmax(etiquetaEstimadaX)
    
        if claseEstimadaX == 0:
            color = 'bx'
        elif claseEstimadaX == 1:
            color = 'rx'
        else:
            color = 'gx'
        plt.plot(punto[0], punto[1], color)
    
    
    """Dibujar para C != 3"""
#    plt.plot(X[0, :], X[1, :], 'o')
#    plt.plot(punto[0], punto[1], 'rx')
#    print(mus[claseEstimadaX])