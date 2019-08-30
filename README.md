# Tarea1RedesNeuronales

## Como correr la tarea?

1) crear un ambiente virtual y activarlo (source venv/bin/activate )
2) instalar los requiriments.txt (pip install -r requirements)
3) correr el archivo analisis con python3 (python3 analisis.py)
4) ver resultados

## Que hay en este repositorio?

1) carpeta NN que contiene la red neuronal y todo lo que usa
2) iris.data, que es un datasets con tipos de flores
3) analisis.py archivo que inicializa una red y se le pasan los datos
de algun dataset. Nos da la matriz de confusion
4) carpeta Tests con tests

## que hay en la carpeta NN?
1) dataSetManipulation.py: archivo que contiene todas las funciones que
manipulan el iris.data.(cargar datos, limpiarlos, revolverlos, hotEncoding, entre otras cosas)
2) funcionesDeActivacion.py: este archivo contiene las funciones de activacion del
enunciado y sus derivadas
3) perseptron.py: perseptron comun y silvestre. como los vistos en clases.
4) redNeuronal.py: objeto redNeuronal que contiene las funciones forward, backpropagation, entre otras.


## Retrospectiva
Quiero destacar que la funcion backPropagation de esta red, es la que vimos en clases, por lo que
es mas facil de programar que otro tipo de modelos pero demora mas en el calculo de resultados.

## analisis
### matris de confusion para 10000 epocas

[14, 0, 5]

[0, 5, 0]

[0, 3, 3]

esta matriz se interpreta de la siguiente forma:

las filas representan las flores que reales.
la fila 0 representa a iris-setosa, las fila a iris-versicolor y la fila 2 a iris-virginica.

las columnas representan exactamente los mismo pero para las flores predichas por la red.

- de esta forma, toda la diagonal son los aciertos ej M[2,2] = aciertos de virginica
- cualquier otro indice fuera de la diagonal es una confusion.


### grafico porcentaje 

Reference-style: 
![alt text][porcentaje]

[porcentaje]: src/porcentajeAcierto.png

vemos como el porcentaje de acierto sube.
A medida que pasan las epocas, el porcentaje empieza a estabilizarse y tender a 100.

(si la foto no se ve, ver src/porcentajeAcierto.png)
### error 

Reference-style: 
![alt text][error]

[error]: src/error.png

vemos como el error baja
realizar 1000 epocas no es suficiente para notar un desenso en el error.
por lo que se realizaron 10.000 epocas para poder mostrar un cambio mas significativo.
(si la foto no se ve, ver src/error.png)

##Extra

- existe un archivo llamado analisisConGraficos.py
este se puede correr si esque se tiene una forma para visualisar graficos
(ej : pycharm)


- si las fotos no se pueden ver en este archivo, estan en la carpeta src

