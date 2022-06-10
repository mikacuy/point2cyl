import plotly.graph_objs as go
import plotly.offline as offline
import torch
import numpy as np
from skimage import measure
import os
import general as utils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle

def plot_surface_2d(decoder,path,epoch, shapename,resolution,mc_value,is_uniform_grid,verbose,save_html,save_ply,overwrite, points=None, with_points=False, latent=None, connected=False):

    filename = '{0}/igr_{1}_{2}'.format(path, epoch, shapename)

    if (not os.path.exists(filename) or overwrite):
        if points is None:
            get_surface_trace_2d(decoder, latent,resolution, mc_value, is_uniform_grid, verbose, filename)
        else:
            get_surface_trace_2d(decoder, latent,resolution, mc_value, is_uniform_grid, verbose, filename, points)
            # get_surface_trace_2d(decoder, latent,1024, mc_value, is_uniform_grid, verbose, filename, points)
            # get_surface_trace_2d(decoder, latent,2048, mc_value, is_uniform_grid, verbose, filename, points)

        return

def draw(coords, values):

    out_points = []
    for ((in_x, in_y), value) in zip(coords, values):
        if abs(value) < 0.01:
            out_points.append(np.array([in_x, in_y]))

    out_points = np.array(out_points)

    return out_points


def get_surface_trace_2d(decoder,latent,resolution,mc_value,is_uniform,verbose,filename, pts=None, sk_contour=False):

    trace = []
    meshexport = None

    if (is_uniform):
        grid = get_grid_uniform_2d(resolution)
    else:
        pass

    z = []

    for i,pnts in enumerate(torch.split(grid['grid_points'],100000,dim=0)):
        if (verbose):
            print ('{0}'.format(i/(grid['grid_points'].shape[0] // 100000) * 100))

        if (not latent is None):
            pnts = torch.cat([latent.expand(pnts.shape[0], -1), pnts], dim=1)
        z.append(decoder(pnts).detach().cpu().numpy())
    
    z = np.concatenate(z,axis=0)
    z = np.reshape(z, (resolution, resolution))


    # if (not (np.min(z) > mc_value or np.max(z) < mc_value)):

    z  = z.astype(np.float64)
    z = np.reshape(z, (resolution * resolution))

    grid['grid_points'] = grid['grid_points'].detach().cpu().numpy()

    ## export to image
    plt.clf()
    fig, ax = plt.subplots()

    cmap = cm.coolwarm

    grid_ = ax.scatter(grid['grid_points'][:, 0], grid['grid_points'][:, 1], c=z, cmap=cmap, vmin= -1.2, vmax= 1.2)

    ### GT
    if pts is not None:
        # pts = pts * (float(resolution)/1.2) 
        ax.scatter(pts[:, 0], pts[:, 1], c='r' , s=1., marker='x')        

    if not sk_contour:
        contour = draw(grid['grid_points'], z)

        if contour.shape[0]>0:
            ax.scatter(contour[:, 0], contour[:, 1], c='b' , s=1.)
    else:
        z = np.reshape(z, (resolution, resolution))
        contours = measure.find_contours(z, level=0.0)

        for contour in contours:
            # Re-align
            contour = contour - resolution/2.
            # contour *= 1.0/256.            
            contour *= 1.2/(resolution/2.) 
            ax.scatter(contour[:, 1], contour[:, 0], c='b' , s=1.)
    
    plt.colorbar(grid_, ax=ax)

    plt.savefig(filename)  
    print(filename)
    plt.close()

    return


def get_grid_uniform_2d(resolution):
    x = np.linspace(-1.2,1.2, resolution)
    y = x

    xx, yy = np.meshgrid(x, y)
    grid_points = utils.to_cuda(torch.tensor(np.vstack([xx.ravel(), yy.ravel()]).T, dtype=torch.float))

    return {"grid_points": grid_points,
            "shortest_axis_length": 2.4,
            "xy": [x, y],
            "shortest_axis_index": 0}