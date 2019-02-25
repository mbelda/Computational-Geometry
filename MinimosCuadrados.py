## Author: MªJose Belda Beneyto
## MINIMOS CUADRADOS 
import numpy as np;
import matplotlib.pyplot as plt;


#Crea 100 datos de dim 2 (1 cosa de 2 filas 100 cols)
#Y las medias
#Necesitamos hacer la mierda esa con mu0 para poder sumar
mu0 = np.array([10,0]);
X0 = np.random.randn(2,100) + mu0[:,np.newaxis];
mu1 = np.array([-10,10]);
X1 = np.random.randn(2,100) + mu1[:,np.newaxis];
mu2 = np.array([-10,-10]);
X2 = np.random.randn(2,100) + mu2[:,np.newaxis];
#Representamos los datos
plt.plot(X0[0,:], X0[1,:], 'o'); #'o' para que pinte bolitas
plt.plot(X1[0,:], X1[1,:], 'o');
plt.plot(X2[0,:], X2[1,:], 'o');

#Creamos las matrices ~X y T de la teoria",
#hstack para apilar matrices horizontalmente (como añadir columnas)
X = np.hstack((X0, X1, X2));
Xg = np.vstack((np.ones((1, X.shape[1])), X));

Unos = np.ones((1,100));
Ceros = np.zeros((1,100));
T1 = np.hstack((Unos,Ceros,Ceros));
T2 = np.hstack((Ceros,Unos,Ceros));
T3 = np.hstack((Ceros,Ceros,Unos));
T = np.vstack((T1,T2,T3));

#Calculamos W gorro
Wg = np.linalg.inv(Xg.dot(Xg.T)).dot(Xg).dot(T.T);
#No trasponemos (X*Xt) a la (-1) pq la mariz deberia ser sim asi que da igual
#Aquí hemos invertido pero esto es ineficiente, deberiamos usar solve y resolver el sistema

X = np.array([1,10,0]);
Wg.T.dot(X);
numpoints = 200;
xs = np.linspace(-15,15,numpoints);
ys = np.linspace(-15,15,numpoints);
Xs, Ys = np.meshgrid(xs, ys);
Xs.flatten();
Ys.flatten();
puntos = np.vstack((np.ones(numpoints**2), Xs.flatten(), Ys.flatten()));
lev = Wg.T.dot(puntos);
print(lev);
#print(Sol.shape); (3,40000)

#Ejercicio\n",
#Este resultado que tenemos pintarlo, de forma que veamos cada punto en que clase esta y ver las fronteras de clasificacion\n",
#Usar la funcion plt.contour
#Transformamos quedandonos con 0, 1 en funcion de la coordenada maxima y usamos contour para pintarlo"
