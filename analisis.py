from NN.dataSetManipulation import *
from NN.funcionesDeActivacion import *
from NN.redNeuronal import *

def analisisIris():

    data=toFloat(loadData('iris.data'))
    shuffle(data)
    xt, xts, yt, yts = divideDataSet(data, 80)

    red=redNeuronal(4,[3,2,3],[tanh(),tanh(),Sigmoid()])
    porcentajeAciertos=[]
    errorCadaEpoca=[]

    #pasaran x epocas
    for j in range(1000):
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

    print("error cada epoca")
    print(errorCadaEpoca)
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





if __name__ == "__main__":
    analisisIris()
