from NN.dataSetManipulation import *
from NN.funcionesDeActivacion import *
from NN.redNeuronal import *

import matplotlib.pyplot as plt

def analisisIris():

    data=toFloat(loadData('iris.data'))
    shuffle(data)
    xt, xts, yt, yts = divideDataSet(data, 80)

    red=redNeuronal(4,[3,2,3],[tanh(),tanh(),Sigmoid()])
    porcentajeAciertos=[]
    errorCadaEpoca=[]
    epocas=1000

    #pasaran x epocas
    for j in range(epocas):
        #esto es una epoca

        for i in range(len(xt)):
            red.train(xt[i],yt[i])

        listaDePredichos=[]
        listaDeReales=[]
        for i in range(len(xts)):

            ypredict=red.preditc(xts[i])
            yreal=red.aprox(yts[i])

            listaDePredichos.append(ypredict)
            listaDeReales.append(yreal)


            red.anotate(ypredict,yreal)

        porcentajeAciertos.append(red.getPorcentajeAccuracy())
        errorCadaEpoca.append(red.errorEpoca(listaDePredichos,listaDeReales))
        red.resetFallosYaciertos()


    print("----------------------------------------")
    print("porcentaje de aciertos")
    print(porcentajeAciertos)
    print("----------------------------------------")
    print(errorCadaEpoca)
    print("error cada epoca")
    print("----------------------------------------")
    print("matriz de confusion")

    matrizconfusion=[[0,0,0],[0,0,0],[0,0,0]]

    for i in range(len(xts)):
        ypredict = red.preditc(xts[i])
        yreal = red.aprox(yts[i])

        ypredict=red.aprox(ypredict)
        yreal=red.aprox(yreal)

        mi=red.getI(ypredict)
        mj=red.getI(yreal)

        matrizconfusion[mi][mj]+=1

    print(matrizconfusion[0])
    print(matrizconfusion[1])
    print(matrizconfusion[2])

    print("----------------------------------------")
    print("ploting")
    plt.plot(range(epocas),porcentajeAciertos, '-r', label='porcentaje acierto', color='#000000')

    plt.title('graficos')
    plt.xlabel('x', color='#1C2833')
    plt.ylabel('y', color='#1C2833')

    plt.legend(loc='upper left')
    plt.grid()
    plt.show()


    plt.plot(range(epocas),errorCadaEpoca, '-r', label='error', color='#000000')

    plt.title('graficos')
    plt.xlabel('x', color='#1C2833')
    plt.ylabel('y', color='#1C2833')

    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    analisisIris()
