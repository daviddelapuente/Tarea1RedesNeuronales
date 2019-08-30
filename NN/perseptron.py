import numpy as np
from NN.funcionesDeActivacion import tanh
class perseptron:

    #la neurona recibe un solo parametro (numero de features)
    #la funcion de activacion es inicialmente es step
    #los features se eligen al azar entre 0 y 1
    #bias es 0
    def __init__(self, n):

        self.W=np.random.rand(n)
        self.bias=0
        self.factivacion=tanh()
        self.output=0
        self.error=0
        self.delta=0
        self.input=[]


    #getters
    def getW(self):
        return self.W
    def getBias(self):
        return self.bias
    def getFactivacion(self):
        return self.factivacion
    def getError(self):
        return self.error
    def getOutput(self):
        return self.output
    def getDelta(self):
        return self.delta
    def getInput(self):
        return self.input

    #devuelve lo que hay en la posicion i del vector input
    def getInputi(self,i):
        return self.input[i]


    #setters
    def setW(self,W):
        self.W=W
    def setBias(self,bias):
        self.bias=bias
    def setFactivacion(self,F):
        self.factivacion=F
    def setWi(self,i,x):
        self.W[i] = x
    def setError(self,e):
        self.error=e
    def setDelta(self,d):
        self.delta=d

    #feed vista en clases
    def feed(self,X):
        #check que X tenga solo valores numericos y que tenga las mismas dimensiones que self.W
        b,e=self.check(X)
        if b:
            try:
                self.input=X
                self.output= self.factivacion.apply(np.sum(np.multiply(self.W,X))+self.getBias())
                return self.output
            except ValueError:
                print("input no es numerico >:C")
        else:
            print(e)

    #checkea que el input tenga la misma dimension que los pesos
    def check(self,X):
        if len(X)!=len(self.getW()):
            return False,"input y pesos tienen dimensiones distintas >:C"
        else:
            return True,""


    #entrena una neurona
    #recibe el input de la nuerona (entradaNeurona),y el resultado esperado
    #modifica los pesos y los bias.
    def train(self,entradaNeurona,desiredOutput):
        realOutput = self.feed(entradaNeurona)
        diff = desiredOutput - realOutput
        lr = 0.1

        for i in range(len(entradaNeurona)):
            self.setWi(i, lr * diff * entradaNeurona[i])

        self.setBias(self.getBias() + lr * diff)
