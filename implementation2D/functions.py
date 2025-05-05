# Imports -- run once
from w2 import BFM
from time import time
import numpy as np
import ot
from matplotlib.colors import LinearSegmentedColormap

from scipy.fftpack import dctn, idctn
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (13, 8)
plt.rcParams['image.cmap'] = 'viridis'

def plotting(dists, dist,name,save_option = False):

  colors = [
    (1.0, 1.0, 1.0),  # White (background)
    (0.0, 0.5, 0.9),  # Very vivid light blue (reduce red and green, max out blue)
    (0.0, 0.3, 0.8),  # Dark blue
    (0.0, 0.0, 0.4)   # Black
  ]
  custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
  vmin, vmax = 0, 160
  plt.imshow(np.sum(dists, axis = 0) + dist, cmap=custom_cmap, origin = 'lower', vmin = vmin, vmax = vmax)

  plt.xticks([0, plt.gca().get_xlim()[1]], ['0', '1'])  # Custom x-axis labels
  plt.yticks([0, plt.gca().get_ylim()[1]], ['0', '1'])  # Custom y-axis labels
  if save_option:
    plt.savefig('%s.jpg'%(name))
  plt.show()

def plotting_mnist(dist, name, save_option=False, ax=None):
    colors = [
        (1.0, 1.0, 1.0),  # White
        (0.0, 0.0, 0.8),  # Light blue
        (0.0, 0.0, 0.7),  # Darker blue
        (0.0, 0.0, 0.6)   # Almost black
    ]
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    vmin, vmax = 0, 70

    if ax is None:
        ax = plt.gca()

    im = ax.imshow(dist, cmap=custom_cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax.set_xticks([0, dist.shape[1]-1])
    ax.set_xticklabels(['0', '1'])
    ax.set_yticks([0, dist.shape[0]-1])
    ax.set_yticklabels(['0', '1'])
    ax.set_title(name)

    # ax.axis('off')  ← disable this to show ticks

    if save_option:
        plt.savefig(f'{name}.jpg')




# Initialize Fourier kernel
def initialize_kernel(n1, n2):
    xx, yy = np.meshgrid(np.linspace(0,np.pi,n1,False), np.linspace(0,np.pi,n2,False))
    kernel = 2*n1*n1*(1-np.cos(xx)) + 2*n2*n2*(1-np.cos(yy))
    kernel[0,0] = 1     # to avoid dividing by zero
    return kernel

# 2d DCT
def dct2(a):
    return dctn(a, norm='ortho')

# 2d IDCT
def idct2(a):
    return idctn(a, norm='ortho')

# Update phi as
#       ϕ ← ϕ + σ Δ⁻¹(ρ − ν)
# and return the error
#       ∫(−Δ)⁻¹(ρ−ν) (ρ−ν)
# Modifies phi and rho
def update_potential(phi, rho, nu, kernel, sigma):
    n1, n2 = nu.shape

    rho -= nu
    workspace = dct2(rho) / kernel
    workspace[0,0] = 0
    workspace = idct2(workspace)

    phi += sigma * workspace
    h1 = np.sum(workspace * rho) / (n1*n2)

    return h1

def grad_norm(rho):
    n2, n1 = np.shape(rho)
    kernel = initialize_kernel(n1,n2)
    workspace = dct2(rho) / kernel
    workspace[0,0] = 0
    workspace = idct2(workspace)

    return np.sum(workspace * rho) / (n1*n2)

def compute_w2(phi, psi, mu, nu):
  n1, n2 = mu.shape
  x, y = np.meshgrid(np.linspace(0,np.pi,n1,False), np.linspace(0,np.pi,n2,False))
  return np.sum(0.5 * (x*x+y*y) * (mu + nu) - nu*phi - mu*psi)/(n1*n2)

# Parameters for Armijo-Goldstein
scaleDown = 0.95
scaleUp   = 1/scaleDown
upper = 0.75
lower = 0.25



def compute_ot(phi, psi, bf,mu, nu, sigma, inner ):
    n2, n1 = np.shape(phi)
    kernel = initialize_kernel(n1, n2)
    rho = np.copy(mu)

    x, y = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1),
                    np.linspace(0.5/n2,1-0.5/n2,n2))
    id = 1/2 * (x**2 + y**2)

    old_w2 = compute_w2(phi, psi, mu, nu)
    for k in range(inner):
        rho = np.zeros((n2,n1))
        bf.pushforward(rho, phi, nu)
        gradSq = update_potential(psi, rho, mu, kernel, sigma)

        bf.ctransform(phi, psi)
        bf.ctransform(psi, phi)

        bf.ctransform(psi, phi)
        bf.ctransform(phi, psi)

        new_w2 = compute_w2(phi, psi, mu, nu)

    return new_w2


def frechet_mean(dists, n_iter,name, plot_option = True,save_option = True, return_option = False,  inner = 1):
  n2, n1 = np.shape(dists[0])
  x, y = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1),
                    np.linspace(0.5/n2,1-0.5/n2,n2))
  n_dist, id, rd = len(dists), 1/2*(x**2 + y**2), dists[0]
  id -= np.mean(id)
  sigma, w2_list = 5e-2 * np.ones(n_dist), np.zeros(n_dist)
  phi, psi = np.array([id] * n_dist), np.array([id] * n_dist)
  tic = time()
  bf = BFM(n1, n2, rd)
  for i in range(n_iter):
    prev_psi = psi
    for j in range(n_dist):
      new_w2 = compute_ot(phi[j], psi[j], bf, rd, dists[j], sigma[j], inner = inner)
      if new_w2 < w2_list[j]:
        sigma[j] *= 0.99
      w2_list[j] = new_w2

    lr = np.exp(-(i+1)/n_iter)
    rho = np.ones_like(rd)
    bf.pushforward(rho, id+ lr*(np.mean(prev_psi,axis=0)-id), rd)
    rd = rho

    if (i+1) % 50 == 0:
      print(f"Number of Iterations : {i+1}")
  toc = time()
  if plot_option:
    plotting(dists, rd,name,save_option = save_option)
  if return_option == True:
    return rd




def frechet_mean_pot(dists, reg,name,plot_option = True,print_option=True, save_option=True, return_option = False):
    num_dist = len(dists)

    weights = np.array([1/num_dist] * num_dist)

    tic = time()
    rd = ot.bregman.convolutional_barycenter2d(dists, reg, weights, numItermax=300,stopThr=0.0)
    toc = time()

    print(f"time spent:{(toc-tic)}s")
    if plot_option:
      plotting(dists, rd,name,save_option = save_option)
    if return_option == True:
        return rd


def frechet_mean_pot_debiased(dists, reg,name,print_option=True, plot_option=True, save_option=True, return_option = False):
    num_dist = len(dists)

    weights = np.array([1/num_dist] * num_dist)

    tic = time()
    rd = ot.bregman.convolutional_barycenter2d_debiased(dists, reg, weights, numItermax = 300,stopThr=0.0)
    toc = time()

    print(f"time spent:{(toc-tic)}s")
    if plot_option:
      plotting(dists, rd,name,save_option = save_option)
    if return_option == True:
        return rd
