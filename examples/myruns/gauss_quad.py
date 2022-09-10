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
			self.gpx, self.gpw, self.J = self.create_GQ_scheme_1d()
		elif self.dim == 2:
			self.gpx, self.gpy, self.gpw, self.J, self.gp = self.create_GQ_scheme_2d() # gp is simply [gpx,gpy]

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
		return gp[:,0], gp[:,1], gpw, J, gp

	def integrate(self, f_eval, f_integrand):
		if self.dim == 1:
			f = f_eval(self.gpx)
		elif self.dim == 2:
			f = f_eval(self.gpx, self.gpy)
		integrand = f_integrand(f)
		gpsum = self.gpw * integrand
		integral = np.sum(gpsum) * self.J
		return integral

	def calc_l2_distance(self, f_eval_1, f_eval_2):
		if self.dim == 1:
			f1 = f_eval_1(self.gpx)
			f2 = f_eval_2(self.gpx)
		elif self.dim == 2:
			f1 = f_eval_1(self.gpx, self.gpy)
			f2 = f_eval_2(self.gpx, self.gpy)
		integrand = (f1 - f2)**2
		gpsum = self.gpw * integrand
		integral = np.sum(gpsum) * self.J
		return np.sqrt(integral)


class GaussQuadratureOnMesh():
	"""docstring for GaussQuadratureOnMesh"""
	def __init__(self, dim, domain=([0.,1.],[0.,1.]), nodes=(11,11), numpt=(2,2)):
		super(GaussQuadratureOnMesh, self).__init__()
		self.dim = dim
		self.xRange = domain[0]
		self.nodesx = nodes[0]
		self.nelmx = self.nodesx - 1
		self.numptx = numpt[0]
		if self.dim >= 2:
			self.yRange = domain[1]
			self.nodesy = nodes[0]
			self.nelmy = self.nodesy - 1
			self.numpty = numpt[1]
		self.create_GQ_mesh_2d()

	def gen_new_scheme(self, numpt=(10,10)):
		self.numptx = numpt[0]
		if self.dim >= 2:
			self.numpty = numpt[1]
		self.create_GQ_mesh_2d()


	def create_GQ_mesh_base(self, range1d, nodes, numpt):
		nel = nodes - 1
		h = (range1d[1] - range1d[0]) / nel

		x = np.linspace(range1d[0], range1d[1], nodes, endpoint=True)
		xl = x[:-1]

		gquad = GaussQuadrature(1, domain=([0., h],), numpt=(numpt,))
		gp = gquad.gpx
		gpw = gquad.gpw

		gp_grid = xl[:, np.newaxis] + gp[np.newaxis, :]

		J = h/2.

		return gp_grid, gpw, J

	def create_GQ_mesh_2d(self):
		gpX, gpwX, Jx = self.create_GQ_mesh_base(self.xRange, self.nodesx, self.numptx)
		gpY, gpwY, Jy = self.create_GQ_mesh_base(self.yRange, self.nodesy, self.numpty)

		self.gpx = np.zeros((self.numpty, self.numptx, self.nelmy, self.nelmx))
		self.gpy = np.zeros((self.numpty, self.numptx, self.nelmy, self.nelmx))
		self.gpw = np.zeros((self.numpty, self.numptx, self.nelmy, self.nelmx))

		for j in range(self.numpty):
			for i in range(self.numptx):
				self.gpx[j, i, :, :] = np.tile(gpX[0:,i], (self.nelmy,1))
				self.gpy[j, i, :, :] = (np.tile(gpY[0:,j], (self.nelmx,1))).T
				self.gpw[j, i, :, :] = (gpwY[j] * gpwX[i])

		self.J = Jx * Jy

		# print("gpx = \n", self.gpx)
		# print("gpy = \n", self.gpy)
		# print("gpwX = \n", gpwX)
		# print("gpwY = \n", gpwY)
		# print("gpw = \n", self.gpw)

	def integrate(self, f_eval, f_integrand):
		f = f_eval(self.gpx, self.gpy)
		integrand = f_integrand(f)
		temp = self.gpw * integrand * self.J
		gpsum = np.sum(np.sum(temp, axis=0), axis=1)
		integral = np.sum(gpsum)
		return integral

	def calc_l2_distance(self, f_eval_1, f_eval_2):
		f1 = f_eval_1(self.gpx, self.gpy)
		f2 = f_eval_2(self.gpx, self.gpy)
		integrand = (f1 - f2)**2
		temp = self.gpw * integrand * self.J
		gpsum = np.sum(np.sum(temp, axis=0), axis=1)
		integral = np.sum(gpsum)
		return np.sqrt(integral)
