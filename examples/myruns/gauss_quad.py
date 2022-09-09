import os
import numpy as np
import numpy.polynomial.legendre as gleg
import matplotlib.pyplot as plt

class GaussQuadrature():
	"""docstring for GaussQuadrature"""
	def __init__(self, dim, domain=([0.,1.],[0.,1.]), numpt=(5,5)):
		super(GaussQuadrature, self).__init__()
		self.dim = dim
		self.xRange = domain[0]
		self.numptx = numpt[0]
		if self.dim >= 2:
			self.yRange = domain[1]
			self.numpty = numpt[1]
		self.create_GQ_scheme()

	def gen_new_scheme(self, numpt=(10,10)):
		self.numptx = numpt[0]
		if self.dim >= 2:
			self.numpty = numpt[1]
		self.create_GQ_scheme()

	def create_GQ_scheme(self):
		if self.dim == 1:
			self.gp, self.gpw, self.J = self.create_GQ_scheme_1d()
		elif self.dim == 2:
			self.gp, self.gpw, self.J = self.create_GQ_scheme_2d()

	def create_GQ_scheme_base(self, xRange, numpt):
		res = gleg.leggauss(numpt)
		gp = res[0]; gpw = res[1]
		
		#translate points
		mid = (xRange[1]+xRange[0]) / 2.
		length_scale = (xRange[1]-xRange[0]) / 2.
		gp = mid + res[0]* length_scale

		J = (xRange[1]-xRange[0]) / 2.
		return gp, gpw, J

	def create_GQ_scheme_1d(self):
		gp, gpw, J = self.create_GQ_scheme_base(self.xRange, self.numptx)
		return gp, gpw, J

	def create_GQ_scheme_2d(self):
		gp_x, gpw_x, Jx = self.create_GQ_scheme_1d()
		gp_y, gpw_y, Jy = self.create_GQ_scheme_1d()
		[XX,YY] = np.meshgrid(gp_x,gp_y)
		WW = gpw_y[:,np.newaxis]*gpw_x[np.newaxis,:]
		gp = np.stack((np.ravel(XX),np.ravel(YY)),1)
		gpw = np.ravel(WW)
		J = Jx*Jy
		return gp, gpw, J

	def integrate(self, f_eval, f_integrand):
		f = f_eval(self.gp)
		integrand = f_integrand(f)
		gpsum = self.gpw * integrand
		integral = np.sum(gpsum) * self.J
		return integral

	def calc_l2_distance(self, f_eval_1, f_eval_2):
		f1 = f_eval_1(self.gp)
		f2 = f_eval_2(self.gp)
		integrand = (f1 - f2)**2
		gpsum = self.gpw * integrand
		integral = np.sum(gpsum) * self.J
		return np.sqrt(integral)

