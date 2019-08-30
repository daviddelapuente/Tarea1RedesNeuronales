from random import shuffle
import numpy as np

#esta funcion lee el dataset que esta en el path y devuelve una matriz con la info del set
#se espera que los datos esten separados por ,
def loadData(path):

    file = open(path)
    data = []
    for line in file:
        row=line.strip("\n").split(",")
        data.append(row)
    return data


#transforma a float todas las columnas menos la ultima
def toFloat(data):
    for i in range(len(data)):
        for j in range(len(data[i]) - 1):
            data[i][j] = float(data[i][j])
    return np.array(data)


#funcion de normalizacion
#decidi que se entragaran los parametros nl y nh porque
#pueden variar segun la funcion que se usan
# si es sigmoid son 0 y 1
# si es tanh son -1 y 1

def normalize(x,nl=-1,nh=1):
    maximos=np.max(x,axis=0)
    minimos=np.min(x,axis=0)
    r=[]
    for i in range(len(x)):
        raux=[]
        for j in range(len(x[i])):
            dl=minimos[j]*1.0
            dh=maximos[j]*1.0
            raux.append(((x[i][j]-dl)*(nh-nl))/(dh-dl)+nl)
        r.append(raux)
    r=np.array(r)

    return r

#esta funcion divide el dataset en una matriz de features (X)
#y un vector con las clases (Y)
def getXandY(data):
    Y=data[:,-1]
    X=data[:,0:-1]
    X=X.astype(float)
    return X,Y

#mappea a [0,..1,...,0]
def hotEncodingIris(data):
    s=set(data)
    ls=len(s)
    m=[]
    for si in s:
        m.append(np.zeros(ls))

    for i in range(ls):
        m[i][i]+=1
    d={}
    count=0
    for i in s:
        d[i]=count
        count+=1

    r=[]

    for y in data:
        i=d[y]
        r.append(m[i])
    return np.array(r)

#devuelve 4 objetos
#el primero es el X de entrenamiento (los features)
#el segundo es el X de teste
#el tercero es el Y de entrenammiento (las clases)
#el cuarto es el y de testeo
#estos parametros se dividen segun un porcentaje p que se entrega (p=80)
#significa 80% de datos para entrenamiento
#ademas entrega los datos normalizados y hotencodeados
def divideDataSet(dataSet,p):

    x,y=getXandY(dataSet)
    x=normalize(x)
    y=hotEncodingIris(y)

    numeroDeDatos=len(x)
    numeroAtributos=len(x[0])

    p80=int(numeroDeDatos*80/100)

    XTraining=[]
    YTraining=[]

    XTesting=[]
    YTesting=[]



    for i in range(numeroDeDatos):
        if i<p80:
            XTraining.append(x[i])
        else:
            XTesting.append(x[i])


    for i in range(numeroDeDatos):
        if i<p80:
            YTraining.append(y[i])
        else:
            YTesting.append(y[i])



    return np.array(XTraining), np.array(XTesting), np.array(YTraining), np.array(YTesting)


#error cuadratico medio
def mse(y1,y2):
    n=len(y1)
    sum=0
    for i in range(n):
        sum+=(y1[i]-y2[i])^2
    return sum/n

