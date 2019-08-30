import unittest
from NN.funcionesDeActivacion import *
class MyTestCase(unittest.TestCase):

    def testStep(self):
        step = Step()
        self.assertTrue(step.apply(3)==1)
        self.assertTrue(step.apply(-3)==0)

    def testTanh(self):
        t=tanh()
        print(t.apply(10))
        print(t.derivative(10))




