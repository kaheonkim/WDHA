# Optimal Transport Barycenter via Nonconvex-Concave Minimax Optimization
link : https://arxiv.org/pdf/2501.14635

**Kaheon Kim, Rentian Yao, Changbo Zhu, and Xiaohui Chen**  
*International Conference on Machine Learning (ICML), 2025*

---

This repository contains the code and experiments for our ICML 2025 paper, _"Optimal Transport Barycenter via Nonconvex-Concave Minimax Optimization."_ 

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{2501.14635,
  title={Optimal Transport Barycenter via Nonconvex-Concave Minimax Optimization},
  author={Kaheon Kim and Rentian Yao and Changbo Zhu and Xiaohui Chen},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```

## Abstract

The optimal transport barycenter (a.k.a. Wasserstein barycenter) is a fundamental notion of averaging that extends from the Euclidean space to the Wasserstein space of probability distributions. Computation of the unregularized barycenter for discretized probability distributions on point clouds is a challenging task when the domain dimension d>1. Most practical algorithms for approximating the barycenter problem are based on entropic regularization. In this paper, we introduce a nearly linear time O(mlogm) and linear space complexity O(m) primal-dual algorithm, the Wasserstein-Descent ℍ˙1-Ascent (WDHA) algorithm, for computing the exact barycenter when the input probability density functions are discretized on an m-point grid. The key success of the WDHA algorithm hinges on alternating between two different yet closely related Wasserstein and Sobolev optimization geometries for the primal barycenter and dual Kantorovich potential subproblems. Under reasonable assumptions, we establish the convergence rate and iteration complexity of WDHA to its stationary point when the step size is appropriately chosen. Superior computational efficacy, scalability, and accuracy over the existing Sinkhorn-type algorithms are demonstrated on high-resolution (e.g., 1024×1024 images) 2D synthetic and real data.

## Dependencies

- numpy
- matplotlib
- scipy
- BFM
- POT(for comparison)
