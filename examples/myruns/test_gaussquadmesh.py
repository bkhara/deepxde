import unittest
import numpy as np
from gauss_quad import GaussQuadratureOnMesh

class TestGaussQuadratureOnMesh(unittest.TestCase):
	"""docstring for TestGaussQuadratureOnMesh"""
	# def __init__(self):
	# 	super(TestGaussQuadratureOnMesh, self).__init__()

	def test_integrate(self):
		Nx = Ny = 16
		gquad = GaussQuadratureOnMesh(2, domain=([0.,1.],[0.,1.]), nodes=(Nx,Ny), numpt=(2,2))

		def f1(x,y): return np.sin(np.pi*x)**2*np.cos(np.pi*y)**2
		# gquad.gen_new_scheme(numpt=(10,10))
		val_c = gquad.integrate(f1, lambda x: x)
		val_a = 0.25
		print("val_c = ", val_c, "val_a = ", val_a, "diff = {:.3e}".format(abs(val_c-val_a)))
		self.assertTrue(abs(val_c-val_a) < 1e-12)

	def test_calc_l2_distance_1(self):
		Nx = Ny = 64
		gquad = GaussQuadratureOnMesh(2, domain=([0.,1.],[0.,1.]), nodes=(Nx,Ny), numpt=(3,3))

		def f1(x,y): return x*y
		def f2(x,y): return x**2*y**2
		val_c = gquad.calc_l2_distance(f1, f2)
		val_a = np.sqrt(1./9. + 1./25. - 1./8.)
		print("val_c = ", val_c, "val_a = ", val_a, "diff = {:.3e}".format(abs(val_c-val_a)))
		self.assertTrue(abs(val_c-val_a) < 1e-12)

		def f1(x,y): return np.sin(np.pi*x)
		def f2(x,y): return np.cos(np.pi*x)
		# gquad.gen_new_scheme(numpt=(10,10))
		val_c = gquad.calc_l2_distance(f1, f2)
		val_a = np.sqrt(1)
		print("val_c = ", val_c, "val_a = ", val_a, "diff = {:.3e}".format(abs(val_c-val_a)))
		self.assertTrue(abs(val_c-val_a) < 1e-1)

		def f1(x,y): return np.sin(np.pi*x)*np.sin(np.pi*y)
		def f2(x,y): return np.cos(np.pi*x)*np.cos(np.pi*y)
		# gquad.gen_new_scheme(numpt=(10,10))
		val_c = gquad.calc_l2_distance(f1, f2)
		val_a = np.sqrt(0.5)
		print("val_c = ", val_c, "val_a = ", val_a, "diff = {:.3e}".format(abs(val_c-val_a)))
		self.assertTrue(abs(val_c-val_a) < 1e-12)

		def f1(x,y): return np.sin(np.pi*x)*np.sin(np.pi*y)
		def f2(x,y): return np.exp(x)*np.exp(y)
		# gquad.gen_new_scheme(numpt=(10,10))
		val_c = gquad.calc_l2_distance(f1, f2)
		val_a = np.sqrt(8.145142998121553314)
		print("val_c = ", val_c, "val_a = ", val_a, "diff = {:.3e}".format(abs(val_c-val_a)))
		self.assertTrue(abs(val_c-val_a) < 1e-12)

if __name__ == '__main__':
	unittest.main()