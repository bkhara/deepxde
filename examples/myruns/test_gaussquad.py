import unittest
import numpy as np
from gauss_quad import GaussQuadrature

class TestGaussQuadrature(unittest.TestCase):
	"""docstring for TestGaussQuadrature"""
	# def __init__(self):
	# 	super(TestGaussQuadrature, self).__init__()

	def test_integrate(self):
		gquad = GaussQuadrature(2, domain=([0.,1.],[0.,1.]), numpt=(5,5))

		def f1(x): return np.sin(np.pi*x[:,0])**2*np.cos(np.pi*x[:,1])**2
		gquad.gen_new_scheme(numpt=(10,10))
		val_c = gquad.integrate(f1, lambda x: x)
		val_a = 0.25
		# print("val_c = ", val_c, "val_a = ", val_a, "diff = {:.3e}".format(abs(val_c-val_a)))
		self.assertTrue(abs(val_c-val_a) < 1e-12)

	def test_calc_l2_distance_1(self):
		gquad = GaussQuadrature(2, domain=([0.,1.],[0.,1.]), numpt=(5,5))
		
		def f1(x): return x[:,0]*x[:,1]
		def f2(x): return x[:,0]**2*x[:,1]**2
		val_c = gquad.calc_l2_distance(f1, f2)
		val_a = np.sqrt(1./9. + 1./25. - 1./8.)
		print("val_c = ", val_c, "val_a = ", val_a, "diff = {:.3e}".format(abs(val_c-val_a)))
		self.assertTrue(abs(val_c-val_a) < 1e-12)

		def f1(x): return np.sin(np.pi*x[:,0])
		def f2(x): return np.cos(np.pi*x[:,0])
		gquad.gen_new_scheme(numpt=(10,10))
		val_c = gquad.calc_l2_distance(f1, f2)
		val_a = np.sqrt(1)
		print("val_c = ", val_c, "val_a = ", val_a, "diff = {:.3e}".format(abs(val_c-val_a)))
		self.assertTrue(abs(val_c-val_a) < 1e-1)

		def f1(x): return np.sin(np.pi*x[:,0])*np.sin(np.pi*x[:,1])
		def f2(x): return np.cos(np.pi*x[:,0])*np.cos(np.pi*x[:,1])
		gquad.gen_new_scheme(numpt=(10,10))
		val_c = gquad.calc_l2_distance(f1, f2)
		val_a = np.sqrt(0.5)
		print("val_c = ", val_c, "val_a = ", val_a, "diff = {:.3e}".format(abs(val_c-val_a)))
		self.assertTrue(abs(val_c-val_a) < 1e-12)

		def f1(x): return np.sin(np.pi*x[:,0])*np.sin(np.pi*x[:,1])
		def f2(x): return np.exp(x[:,0])*np.exp(x[:,1])
		gquad.gen_new_scheme(numpt=(10,10))
		val_c = gquad.calc_l2_distance(f1, f2)
		val_a = np.sqrt(8.145142998121553314)
		print("val_c = ", val_c, "val_a = ", val_a, "diff = {:.3e}".format(abs(val_c-val_a)))
		self.assertTrue(abs(val_c-val_a) < 1e-12)

if __name__ == '__main__':
	unittest.main()