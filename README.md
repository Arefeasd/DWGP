# Depth Weighted Gaussian Process (DWGP)

This repository contains the official implementation of the **Depth Weighted Gaussian Process (DWGP)** model introduced in our paper:

> **Depth Weighted Gaussian Process**  
> July 2025  
> Arefeh Asadi

##  Overview

Gaussian Processes (GPs) are powerful nonparametric models for regression and uncertainty quantification. However, their sensitivity to outliers—due to the Gaussian noise assumption—limits their applicability in real-world scenarios.

In this project, we propose a novel robust extension of Gaussian Processes:  
**Depth Weighted Gaussian Process (DWGP)** — a model that incorporates data-depth-based weights into the GP likelihood to reduce the influence of outlying observations.

The DWGP method:
- Computes **local Mahalanobis depth** for each observation.
- Transforms depth into **observation-specific weights** using a quasi-robust weighting function.
- Optimizes GP hyperparameters under a **weighted negative log marginal likelihood**.
- Offers robustness without sacrificing predictive accuracy.
