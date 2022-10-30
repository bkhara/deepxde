"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import os
import sys
sys.path.insert(0, '/work/baskarg/bkhara/deepxde-multi-vers/deepxde-bkhara')
import numpy as np
import amathlib.integration as ami
import deepxde as dde
from deepxde.backend import tf
import utilities as mutils
import json
import pickle

backend_name = os.environ["DDE_BACKEND"]

pi = np.pi

def calc_l2_distance_v2(gquad, f_eval_1, f_eval_2):
    '''
    The signature of both f_eval_1 and f_eval_2 is given by: f(X) -> u
    where X =[x], or [x,y] or [x,y,z]
    '''
    if gquad.dim == 1:
        gp = gquad.gpx[:, np.newaxis]
    elif gquad.dim == 2:
        gp = np.stack((gquad.gpx, gquad.gpy),1)
    f1 = f_eval_1(gp).squeeze()
    f2 = f_eval_2(gp).squeeze()
    integrand = (f1 - f2)**2
    gpsum = gquad.gpw * integrand
    integral = np.sum(gpsum) * gquad.J
    return np.sqrt(integral)

def y_analytical(x):
    return np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:])

def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    return -dy_xx - dy_yy - 2.*pi**2*tf.sin(pi * x[:, 0:1]) * tf.sin(pi * x[:, 1:])


def boundary(_, on_boundary):
    return on_boundary

# setup directory
model_dir = mutils.get_next_run('./p2d_dirichlet')
print("model_dir = ", model_dir)
os.makedirs(model_dir)
# =================================================================================
# parameters
nx = ny = 64
LR = 1e-3
epochs = 10000
# hidden layer config, W=width, D=depth
HLW = 10
HLD = 3
# hidden_layers = [HLW] * HLD
hidden_layers = [8,12,7]

# geom = dde.geometry.Polygon([[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]])
# geom = dde.geometry.Polygon([[0, 0], [1, 0], [1, 1], [1, 0]])
geom = dde.geometry.Rectangle([0, 0], [1, 1])
bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

data = dde.data.PDE(geom, pde, bc, num_domain=nx*ny, num_boundary=2*(nx+ny), num_test=1500)
# data = dde.data.PDE(geom, pde, bc, num_domain=nx*ny, num_boundary=2*(nx+ny), num_test=1500, train_distribution='uniform')
# data = dde.data.PDE(geom, pde, bc, num_domain=32*32, num_boundary=128, num_test=1500, train_distribution='uniform')
net = dde.nn.FNN([2] + hidden_layers + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)

loss_weights = [1., 50.]

model.compile("adam", lr=LR, loss_weights=loss_weights)
losshistory, train_state = model.train(iterations=epochs)

# error calculation
gquad = ami.GaussQuadrature(2, domain=([0.,1.],[0.,1.]), numpt=(20,20))
# el2 = gquad.calc_l2_distance_v2(model.predict, y_analytical)
el2 = calc_l2_distance_v2(gquad, model.predict, y_analytical)
print("||e|| = {:5e}".format(el2))

model.compile("L-BFGS", loss_weights=loss_weights)
losshistory, train_state = model.train()

model.save(save_path=os.path.join(model_dir, 'p2d_d'), verbose=1)
dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir=model_dir)

# error calculation
# el2 = gquad.calc_l2_distance_v2(model.predict, y_analytical)
el2 = calc_l2_distance_v2(gquad, model.predict, y_analytical)
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

# plot
X_query = mutils.query_points_structured_grid(nx, ny)
y_pred = model.predict(X_query)
y_true = y_analytical(X_query)
diff = y_pred - y_true
np.save(os.path.join(model_dir, 'ypred_{}x{}.npy'.format(nx, ny)), y_pred)

# =================================================================================
# post processing
mutils.plot_contour(y_pred, y_true, model_dir, dump_dict)

# =================================================================================
# POST PROCESSING AND EXPERIMENTS
# =================================================================================

# print(model.data.train_x)
num_boundary = model.data.num_boundary
if backend_name == "tensorflow":
    np.savetxt(os.path.join(model_dir, 'x_bo.txt'), model.data.train_x[0:num_boundary,:], delimiter=',')
    np.savetxt(os.path.join(model_dir, 'x_domain.txt'), model.data.train_x[num_boundary:,:], delimiter=',')


random_prediction = False
experimental = False

if random_prediction:
    # calc residual of some random points
    x = np.array([
            [0.90668415, 0.37842557],
            [0.97757273, 0.81089117],
            [0.07580340, 0.44260217],
            [0.49999999, 0.49999999]
            ])
    u = model.predict(x, operator=None)
    res = model.predict(x, operator=pde)
    print("x = \n", x)
    print("u = \n", u)
    print("residual = \n", res)

if experimental:
    # uu = model.net(x)
    # res = pde(x, uu)
    # print("residual = \n", res)

    if backend_name == "tensorflow.compat.v1":
        op = pde(model.net.inputs, model.net.outputs)
        feed_dict = model.net.feed_dict(False, x)
        y = model.sess.run(op, feed_dict=feed_dict)
        print("res = ", y)

    if backend_name == "tensorflow":
        # print(model.net.trainable_weights)
        # model.net.save_weights(os.path.join(model_dir, 'p2d_d_m'))

        # print(model.net.trainable_weights)
        # model.net.load_weights('/work/baskarg/bkhara/deepxde-multi-vers/deepxde-bkhara/examples/myruns/p2d_dirichlet/run_147/p2d_d_m')
        # print(model.net.trainable_weights)
        wt_dict = {}
        for i, w in enumerate(model.net.trainable_weights):
            wt_type = 'w' if i%2 == 0 else 'b'
            wt_dict[wt_type+str(i//2)] = w.numpy()
        with open(os.path.join(model_dir, 'wt_dict.pickle'), 'wb') as handle:
            pickle.dump(wt_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print(wt_dict)

