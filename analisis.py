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
    for j in range(10):
        #esto es una epoca

        for i in range(len(xt)):
            red.train(xt[i],yt[i])

        aciertos=0
        fallos=0
        predicciones=[]
        for i in range(len(xts)):

            ypredict=red.preditc(xts[i])
            predicciones.append(ypredict)

            yreal=red.aprox(yts[i])
            mi=getI(ypredict)
            mj=getI(yreal)
            if mi==mj:
                aciertos+=1
            else:
                fallos+=1

        if j%100==0:
            porcentajeAciertos.append(100*aciertos/(aciertos+fallos))


    print(porcentajeAciertos)

def getI(y):
    i=0
    for j in range(len(y)):
        if y[j]==1:
            return i
        else:
            i+=1


if __name__ == "__main__":
    analisisIris()
