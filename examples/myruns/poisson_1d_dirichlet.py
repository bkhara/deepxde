"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import os
import sys
sys.path.insert(0, '/work/baskarg/bkhara/deepxde-multi-vers/deepxde-bkhara')
import numpy as np
import amathlib.integration as ami
import utilities as mutils
import json
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
# Import tf if using backend tensorflow.compat.v1 or tensorflow
from deepxde.backend import tf
# Import torch if using backend pytorch
# import torch
# Import paddle if using backend paddle
# import paddle


def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    # Use tf.sin for backend tensorflow.compat.v1 or tensorflow
    return -dy_xx - np.pi ** 2 * tf.sin(np.pi * x)
    # Use torch.sin for backend pytorch
    # return -dy_xx - np.pi ** 2 * torch.sin(np.pi * x)
    # Use paddle.sin for backend paddle
    # return -dy_xx - np.pi ** 2 * paddle.sin(np.pi * x)


def boundary(x, on_boundary):
    return on_boundary


def func(x):
    return np.sin(np.pi * x)


# setup directory
model_dir = mutils.get_next_run('./p1d_dirichlet')
print("model_dir = ", model_dir)
os.makedirs(model_dir)
# =================================================================================
# parameters
nx = 128
LR = 1e-3
epochs = 50000
# hidden layer config, W=width, D=depth
HLW = 100
HLD = 10

geom = dde.geometry.Interval(-1, 1)
bc = dde.icbc.DirichletBC(geom, func, boundary)
data = dde.data.PDE(geom, pde, bc, nx, 2, solution=func, num_test=100, train_distribution='uniform')

layer_size = [1] + [HLW] * HLD + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=epochs)
# Optional: Save the model during training.
# checkpointer = dde.callbacks.ModelCheckpoint(
#     "model/model", verbose=1, save_better_only=True
# )
# Optional: Save the movie of the network solution during training.
# ImageMagick (https://imagemagick.org/) is required to generate the movie.
# movie = dde.callbacks.MovieDumper(
#     "model/movie", [-1], [1], period=100, save_spectrum=True, y_reference=func
# )
# losshistory, train_state = model.train(iterations=10000, callbacks=[checkpointer, movie])
model.save(save_path=os.path.join(model_dir, 'p1d_d'), verbose=1)
dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir=model_dir)

# Optional: Restore the saved model with the smallest training loss
# model.restore(f"model/model-{train_state.best_step}.ckpt", verbose=1)
# Plot PDE residual
x = geom.uniform_points(100, True)
y = model.predict(x, operator=None)
y_exact = func(x)
plt.figure()
plt.plot(x, y, 'b-')
plt.plot(x, y_exact, 'r*--')
plt.xlabel("x")
plt.ylabel("Contours")
plt.savefig(os.path.join(model_dir, 'contours.png'))
plt.close()
y = model.predict(x, operator=pde)
plt.figure()
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("PDE residual")
# plt.show()
plt.savefig(os.path.join(model_dir, 'residual.png'))

# error calculation
gquad = ami.GaussQuadrature(1, domain=([0.,1.],), numpt=(20,))
el2 = gquad.calc_l2_distance_v2(model.predict, func)
print("||e|| = {:5e}".format(el2))

