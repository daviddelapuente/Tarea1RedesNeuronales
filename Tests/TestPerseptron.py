import unittest
from NN.perseptron import *

class MyTestCase(unittest.TestCase):

    def testPesos(self):
        p=perseptron(1)
        p.setW([10])
        self.assertEqual([10],p.getW())

        p.setW([10,20,40])
        self.assertEqual([10,20,40],p.getW())

        p.setW([0,0,0])
        self.assertEqual([0,0,0],p.getW())

    def testSesgos(self):
        p=perseptron(1)
        self.assertEqual(0,p.getBias())
        p.setBias(100)
        self.assertEqual(100,p.getBias())

    def testFeed(self):
        p=perseptron(1)
        p.setW([10])
        self.assertEqual(20,p.feed(2))

        p.setBias(-5)
        p.setW([10,20,-10])
        self.assertEqual(200,p.feed([0.5,10,0]))

    def testFactivacion(self):
        p=perseptron(1)
        f=p.getFactivacion()
        self.assertEqual(f.apply(-10),0)



