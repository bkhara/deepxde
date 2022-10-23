"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import sys
sys.path.insert(0, '/work/baskarg/bkhara/deepxde-multi-vers/deepxde-bkhara')
import amathlib.integration as ami
import deepxde as dde

def y_analytical(x):
    # return np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:])
    return 0.

def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    return -dy_xx - dy_yy - 1


def boundary(_, on_boundary):
    return on_boundary


geom = dde.geometry.Polygon([[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]])
bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

data = dde.data.PDE(geom, pde, bc, num_domain=1200, num_boundary=120, num_test=1500)
net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)

model.compile("adam", lr=0.001)
model.train(iterations=50000)
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)


# error calculation
gquad = ami.GaussQuadrature(2, domain=([0.,1.],[0.,1.]), numpt=(5,5))
el2 = gquad.calc_l2_distance_v2(model.predict, y_analytical)
print("||e|| = {:5e}".format(el2))
