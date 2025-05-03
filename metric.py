from w2 import BFM
from time import time
import numpy as np
import numpy.ma as ma
from scipy.fftpack import dctn, idctn
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (13, 8)
plt.rcParams['image.cmap'] = 'viridis'
numIters = 100


def initialize_kernel(n1, n2):
    xx, yy = np.meshgrid(np.linspace(0,np.pi,n1,False), np.linspace(0,np.pi,n2,False))
    kernel = 2*n1*n1*(1-np.cos(xx)) + 2*n2*n2*(1-np.cos(yy))
    kernel[0,0] = 1     # to avoid dividing by zero
    return kernel


def dct2(a):
    return dctn(a, norm='ortho')


def idct2(a):
    return idctn(a, norm='ortho')


def update_potential(phi, rho, nu, kernel, sigma):
    n1, n2 = nu.shape

    rho -= nu
    workspace = dct2(rho) / kernel
    workspace[0,0] = 0
    workspace = idct2(workspace)

    phi += sigma * workspace
    h1 = np.sum(workspace * rho) / (n1*n2)

    return h1


def compute_w2(phi, psi, mu, nu, x, y):
    n1, n2 = np.shape(phi)
    return np.sum(0.5 * (x*x+y*y) * (mu + nu) - nu*phi - mu*psi)/(n1*n2)


scaleDown = 0.95
scaleUp   = 1/scaleDown
upper = 0.75
lower = 0.25

def stepsize_update(sigma, value, oldValue, gradSq):
    diff = value - oldValue

    if diff > gradSq * sigma * upper:
        return sigma * scaleUp
    elif diff < gradSq * sigma * lower:
        return sigma * scaleDown
    return sigma



def compute_ot(mu, nu, bf, sigma, numIters):
    n2,n1 = np.shape(mu)
    x, y = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1), np.linspace(0.5/n2,1-0.5/n1,n2))

    phi = 0.5 * (x*x + y*y)
    psi = 0.5 * (x*x + y*y)


    kernel = initialize_kernel(n1, n2)
    rho = np.copy(mu)

    oldValue = compute_w2(phi, psi, mu, nu, x, y)

    for k in range(numIters):

        gradSq = update_potential(phi, rho, nu, kernel, sigma)

        bf.ctransform(psi, phi)
        bf.ctransform(phi, psi)

        value = compute_w2(phi, psi, mu, nu, x, y)
        sigma = stepsize_update(sigma, value, oldValue, gradSq)
        oldValue = value

        bf.pushforward(rho, phi, nu)

        gradSq = update_potential(psi, rho, mu, kernel, sigma)

        bf.ctransform(phi, psi)
        bf.ctransform(psi, phi)

        bf.pushforward(rho, psi, mu)

        value = compute_w2(phi, psi, mu, nu, x, y)
        sigma = stepsize_update(sigma, value, oldValue, gradSq)
        oldValue = value

        # if (k+1) % 500 == 0:
        #     print(f'iter {k+1:4d},   W2 value: {value:.6e},   H1 err: {gradSq:.2e}')
    return value

def w2dist(mu, nu):
    n1,n2 = np.shape(mu)
    numIters = 500
    # Initial step size
    sigma = 4/np.maximum(mu.max(), nu.max())

    tic = time()

    # Initialize BFM method
    bf = BFM(n1, n2, mu)
    w2 = compute_ot(mu, nu, bf, sigma, numIters)

    toc = time()
    return w2

def avgw2(mu, mu_dist):
  dist_list = []
  for k in range(len(mu)):
    dist_list.append(w2dist(mu[k], mu_dist))
  return np.mean(dist_list)


def l2dist(phi,psi):
  n2, n1 = np.shape(phi)
  PHI = np.sum((phi-psi)**2/n1/n2)
  return np.sqrt(PHI)
