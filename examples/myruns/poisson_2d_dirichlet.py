from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import numpy.polynomial.legendre as gleg
import matplotlib.pyplot as plt
from gauss_quad import GaussQuadrature
import utilities as mutils
import pickle
import json

# os.environ["DDE_BACKEND"] = "tensorflow"
import deepxde as dde
from deepxde.backend import tf
# tf.keras.backend.set_floatx('float64')

def forcing(x):
    return 2.0 * np.pi ** 2 * np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:])

def pde(x, y):
    print("size of x = ", x.shape)
    dy_x = tf.gradients(y, x)[0]
    dy_x, dy_y = dy_x[:, 0:1], dy_x[:, 1:]
    dy_xx = tf.gradients(dy_x, x)[0][:, 0:1]
    dy_yy = tf.gradients(dy_y, x)[0][:, 1:]
    return -dy_xx - dy_yy - 2.0 * np.pi ** 2 * tf.sin(np.pi * x[:, 0:1]) * tf.sin(np.pi * x[:, 1:])

def boundary(x, on_boundary):
    return on_boundary

def y_analytical(x):
    return np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:])


# =================================================================================
# print("GPU Available? = ", tf.test.is_gpu_available())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# setup directory
model_dir = mutils.get_next_run('./p2d_dirichlet')
print("model_dir = ", model_dir)
os.makedirs(model_dir)
# =================================================================================
# parameters
nx = ny = 32
LR = 1e-3
epochs = 20000
# hidden layer config, W=width, D=depth
HLW = 100
HLD = 8

# =================================================================================
# PINN magic
geom = dde.geometry.Rectangle([0, 0], [1, 1])
bc = dde.DirichletBC(geom, lambda x: 0., boundary)

data = dde.data.PDE(geom, pde, bc, num_domain=nx*ny, num_boundary=2*(nx+ny), num_test=1500, train_distribution='uniform')
net = dde.maps.FNN([2] + [HLW] * HLD + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)

model.compile("adam", lr=LR)
losshistory, train_state = model.train(epochs=epochs)
model.compile("L-BFGS-B")
losshistory, train_state = model.train()

model.save(save_path=os.path.join(model_dir, 'p2d_d'), verbose=1)
dde.saveplot(losshistory, train_state, issave=True, isplot=False)
# =================================================================================

# error calculation
gquad = GaussQuadrature(2, domain=([0.,1.],[0.,1.]), numpt=(5,5))
el2 = mutils.calc_l2_distance(gquad, model.predict, y_analytical)
print("||e|| = {:5e}".format(el2))

# save parameters
dump_dict = {
        'nx': nx,
        'ny': ny,
        'HLW': HLW,
        'HLD': HLD,
        'LR': LR,
        'epochs': epochs,
        'el2': el2
        }
with open(os.path.join(model_dir, 'data.json'), 'w') as f:
    json.dump(dump_dict, f, indent="\t")

X_query = mutils.query_points_structured_grid(nx, ny)
y_pred = model.predict(X_query)
y_true = y_analytical(X_query)
diff = y_pred - y_true
np.save(os.path.join(model_dir, 'ypred_{}x{}.npy'.format(nx, ny)), y_pred)

# =================================================================================
# post processing
mutils.plot_contour(y_pred, y_true, model_dir, dump_dict)


