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