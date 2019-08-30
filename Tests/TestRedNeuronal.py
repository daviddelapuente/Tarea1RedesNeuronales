import unittest
from NN.redNeuronal import *
from NN.funcionesDeActivacion import *

class MyTestCase(unittest.TestCase):

    def testBiasAndPesosDeRed(self):
        r=redNeuronal(5,[3,2,1],[tanh,tanh,Sigmoid])

        self.assertEqual(len(r.getRed()[0][0].getW()), 5)
        self.assertEqual(len(r.getRed()[0][2].getW()), 5)
        self.assertEqual(len(r.getRed()[1][1].getW()), 3)
        self.assertEqual(len(r.getRed()[2][0].getW()), 2)

        r.setBias([1,2,3],0)

        self.assertEqual(r.getRed()[0][0].getBias(),1)
        self.assertEqual(r.getRed()[0][1].getBias(),2)

        r.setBias([5,6],1)

        self.assertEqual(r.getRed()[1][0].getBias(),5)
        self.assertEqual(r.getRed()[1][1].getBias(),6)

        r.setPesos([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]],0)

        self.assertEqual(len(r.getRed()[0][0].getW()), 5)
        self.assertEqual(len(r.getRed()[0][2].getW()), 5)

        self.assertEqual(r.getRed()[0][1].getW()[3],9)


    def testForward(self):
        r = redNeuronal(5, [3, 2, 2], [tanh(), tanh(), Sigmoid()])
        print(r.forward([1,1,1,1,1]))
        r.train([1,1,1,1,1],np.array([0,1]))
        print(r.forward([1,1,1,1,1]))




