import numpy as np

class Sigmoid:
    def apply(self,x):
        return 1/(1+np.exp(-x))

    def derivative(self,x):
        return self.apply(x)*(1-self.apply(x))

class Step:
    def apply(self,x):
        if x>=0:
            return 1
        else:
            return 0

    def derivative(self,x):
        if x==0:
            print("Error, Step no tiene derivada en 0")
            return False
        else:
            return 0

class tanh:
    def apply(self,x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    def derivative(self,x):
        t=self.apply(x)

        return 1-t*t