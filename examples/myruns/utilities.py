import os
import numpy as np
from gauss_quad import GaussQuadrature
import matplotlib.pyplot as plt

# def l2_err(model, yfunc, gpoints, Nmesh):
#     ypinn_gp = model.predict(gpoints)
#     yexact_gp = yfunc(gpoints)
#     diff_gp = (yexact_gp - ypinn_gp)**2
#     diff_gp = np.reshape(diff_gp, (4,(Nmesh-1),(Nmesh-1)))
#     diff_gp_sum = np.sum(diff_gp)
#     el2 = (diff_gp_sum)*(0.5/Nmesh)*(0.5/Nmesh)
#     return np.sqrt(el2)

# def l2_err_2(model, yfunc, gpoints, gpweights, J):
#     ypinn_gp = model.predict(gpoints)
#     yexact_gp = yfunc(gpoints)
#     diff_gp = gpweights*(yexact_gp - ypinn_gp)**2
#     diff_gp_sum = np.sum(diff_gp)
#     el2 = (diff_gp_sum)*J
#     return np.sqrt(el2)

def get_next_run(output_path):
    idx = 0
    path = os.path.join(output_path, "run_{:03d}".format(idx))
    while os.path.exists(path):
        idx += 1
        path = os.path.join(output_path, "run_{:03d}".format(idx))
    return path

def query_points_structured_grid(nx, ny):
    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    X_query = (np.vstack((np.ravel(xx), np.ravel(yy)))).T
    return X_query


def calc_l2_distance(gquad, f_eval_1, f_eval_2):
    gp = np.stack((gquad.gpx, gquad.gpy),1)
    f1 = f_eval_1(gp)
    f2 = f_eval_2(gp)
    integrand = (f1 - f2)**2
    gpsum = gquad.gpw * integrand
    integral = np.sum(gpsum) * gquad.J
    return np.sqrt(integral)

def plot_contour(y_pred, y_true, model_dir, dump_dict):
    nx = dump_dict['nx']
    ny = dump_dict['ny']
    LR = dump_dict['LR']
    HLW = dump_dict['HLW']
    HLD = dump_dict['HLD']

    y_pred = np.reshape(y_pred, (ny, nx))
    y_true = np.reshape(y_true, (ny, nx))
    diff = y_pred - y_true

    fig, axs = plt.subplots(1, 3, figsize=(3*4,2), subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    im0 = axs[0].imshow(y_pred,cmap='jet')
    # fig.colorbar(im0, ax=axs[0], ticks=[0.0, 4.0, 8.0, 12.0, 16.0, 20.0])
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(y_true,cmap='jet')
    fig.colorbar(im1, ax=axs[1])
    im = axs[2].imshow(diff,cmap='jet')
    fig.colorbar(im, ax=axs[2])
    fig.suptitle("Nx = {}, Ny = {}, LR = {:.1e}, FNN-HLayers = {}x{}".format(nx, ny, LR, HLW, HLD), fontsize=8) 
    plt.savefig(os.path.join(model_dir, 'contours.png'))
    plt.close('all')