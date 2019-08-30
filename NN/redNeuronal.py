from NN.perseptron import *

class redNeuronal:

    #nEntradas=numero de entradas (features) (es un vector)
    # nNeuronas= lista de los numeros de neuronas de cada capa
    # funciones = lista de funciones
    # se omitieron algunos inputs que de alguna forma estan contenidos en los que
    # recibe este constructor. por ejemplo numero de valores de salida, debe ir includo
    # en el argumento nNeuronas (en la ultima posicion)
    def __init__(self,nEntradas,nNeuronas,funciones,learningRate=0.001):
        self.nEntradas=nEntradas
        self.learningRate=learningRate
        self.funciones=funciones
        self.aciertos=0
        self.fallos=0

        largoPesos=[]
        largoPesos.append(nEntradas)

        for i in nNeuronas:
            largoPesos.append(i)

        self.red=[]
        for i in range(len(nNeuronas)):
            capaAux=[]
            for j in range(nNeuronas[i]):
                capaAux.append(perseptron(largoPesos[i]))
                capaAux[j].setFactivacion(funciones[i])

            self.red.append(capaAux)


    #devuelve la red
    def getRed(self):
        return self.red

    #cambia el bias de una capa (recibe un vector de bias para la capa)
    def setBias(self,tBias,capa):
        c=0
        for p in self.red[capa]:
            p.setBias(tBias[c])
            c+=1

    #cambia los pesos de una capa
    def setPesos(self,tPesos,capa):
        c=0
        for p in self.red[capa]:
            p.setW(tPesos[c])
            c+=1

    #recibe un vector de tamaño del tamaño especificado en el atributo nEntradas del constructor

    def forward(self,input):
        b,e=self.check(input)
        #checkea que los inputs sean numericos
        if b:
            try:
                result=input

                for capa in self.red:
                    h=[]
                    for p in capa:
                        h.append(p.feed(result))
                    result=h

                return result
            except ValueError:
                print("input no es numerico >:C")
        else:
            print(e)

    #checkque las dimensiones
    def check(self,input):
        if len(input)!=self.nEntradas:
            return False,"input y pesos tienen dimensiones distintas >:C"
        else:
            return True,""


    #backpropagation, se le pasa un yOutput calculado normalmente con forward
    #y se le pasa el Y esperado.
    #se hizo sin matrices, por lo que el tiempo computacional de este algoritmo es ineficiente.
    def backpropagation(self,yOutput,yExpected):
        error=yExpected-yOutput
        c=0
        for p in self.red[len(self.red)-1]:
            p.setError(error[c])
            p.setDelta(p.getError()*p.getFactivacion().derivative(p.getOutput()))
            c+=1

        aux=len(self.red)-2
        for i in range(aux):
            capa=self.red[aux-i]
            c=0
            for p in capa:
                p.setError(self.getErrorCapa(aux-i+1,c))
                p.setDelta(p.getError()*p.getFactivacion().derivative(p.getOutput()))
                c += 1

    #calcula el error de una capa para una neurona que esta antes de esa capa (de izquierda a derecha)
    #el error es la suma de los delta de cada neurona de la capa, multiplicado por el peso de la neurona
    #que esta antes.
    def getErrorCapa(self,capa,i):
        sum=0
        for p in self.red[capa]:

            sum+=p.getDelta()*p.getW()[i]
        return sum

    #esto es lo que ocurre dsps del backpropagation
    #se ajustan los pesos y los bias segun el algoritmo visto en clases
    def ajustarPesosYbias(self):
        for capa in self.red:
            for p in capa:
                p.setBias(p.getBias()+self.learningRate*p.getDelta())
                c=0
                for w in p.getW():
                    p.setWi(c,w+self.learningRate*p.getDelta()*p.getInputi(c))
                    c+=1

    #es un forward seguido de un backpropagation y alfinal ajusta pesos y bias.
    def train(self,input,esperado):
        y=self.forward(input)
        self.backpropagation(y,esperado)
        self.ajustarPesosYbias()

    #calcula el forward de un input X y hace argmax
    def preditc(self,x):
        return self.aprox(self.forward(x))

    #funciona como argmax
    #recibe un vector con numeros y devuelve un vector
    #con un 1 en la posicion del maximo y 0's en las demas posiciones
    def aprox(self,x):
        i = 0

        for j in range(len(x)):
            if x[j] >= x[i]:
                i = j

        r = []
        for j in range(len(x)):
            if j == i:
                r.append(1)
            else:
                r.append(0)
        return r

    #argmax
    def getI(self,y):
        i = 0
        for j in range(len(y)):
            if y[j] == 1:
                return i
            else:
                i += 1


    #cuenta cuantas predicciones tuvo buena y cuantas malas
    def anotate(self,yp,yr):
        mi = self.getI(yp)
        mj = self.getI(yr)
        if mi == mj:
            self.aciertos += 1
        else:
            self.fallos += 1

    def resetFallosYaciertos(self):
        self.fallos=0
        self.aciertos=0

    def getPorcentajeAccuracy(self):
        return 100*self.aciertos/(self.aciertos+self.fallos)

    def errorEpoca(self,lyp,lyr):
        lyp=np.array(lyp)
        lyr=np.array(lyr)
        return np.sum((0.5*(lyp-lyr)**2).mean(axis=1))/len(lyp)