import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.stats import norm
from scipy.integrate import quad # This function is used for numerical integration of a function over a given interval.
from scipy.optimize import differential_evolution
import scipy.linalg
from sklearn.covariance import MinCovDet
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.spatial.distance import cdist

def mahalanobis_depth(data, point):
    mu = np.mean(data)
    sigma = np.std(data)
    if sigma < 1e-6:
        sigma = 1e-6
    depth = 1 / (1 + np.abs(point - mu) / sigma)
    return depth

def weight(z, gamma, lamda):
    w = (1-gamma)* np.exp(-lamda*(z**2)) + gamma
    return w

def mahalanobis_distance(x, data, cov_inv):
    """ Compute the Mahalanobis distance between x and a dataset. """
    return np.array([mahalanobis(x, d, cov_inv) for d in data])



from sklearn.covariance import MinCovDet

def compute_robust_weights(X, y, r, s, gamma, eta=0.5):
    N, D = X.shape
    weights = np.zeros(N)

    covar = np.cov(X, rowvar=False)
    if covar.ndim == 0:
        covar = np.array([[covar]])
    if np.linalg.cond(covar) > 1e10:
        covar += np.eye(covar.shape[0]) * 1e-5
    cov_inv = np.linalg.inv(covar)

    for i in range(N):
        distances = mahalanobis_distance(X[i], X, cov_inv)
        neighborhood_idx = np.where(distances <= r)[0]

        if len(neighborhood_idx) < s:
            weights[i] = eta
            continue

        yi_neighbors = y[neighborhood_idx].reshape(-1, 1)

        if (
            np.any(np.isnan(yi_neighbors)) or
            np.any(np.isinf(yi_neighbors)) or
            np.std(yi_neighbors) < 1e-6 or
            yi_neighbors.shape[0] < 2
        ):
            weights[i] = eta
            continue

        try:
            mcd = MinCovDet(support_fraction=0.5).fit(yi_neighbors)  # ✅ این خط مهمه
            mu_i = mcd.location_[0]
            v_i = mcd.covariance_[0, 0]

            if not np.isfinite(mu_i) or not np.isfinite(v_i) or v_i < 1e-10:
                weights[i] = eta
                continue

            z_i = (y[i] - mu_i) / np.sqrt(v_i + 1e-8)
            z_i = np.atleast_1d(z_i)[0]
            weights[i] = (1 - gamma) * np.exp(-z_i**2) + gamma

        except Exception as e:
            weights[i] = eta

    weights = np.clip(weights, 1e-3, 10)
    return weights



def exponential_kernel(X1, X2, theta):
    """
    Computes the squared exponential (RBF) kernel matrix using anisotropic (per-dimension) length-scales.
    
    Args:
        X1: (N, D) NumPy array of input points.
        X2: (M, D) NumPy array of input points.
        theta: (D,) NumPy array of positive values — inverse squared length-scales.
               (Note: If using actual length-scales ℓ_i, then theta_i = 1 / ℓ_i^2)

    Returns:
        (N, M) Kernel matrix.
    """
    # Scale inputs using sqrt of theta to apply per-dimension weighting
    X1_scaled = X1 * np.sqrt(theta)
    X2_scaled = X2 * np.sqrt(theta)

    # Compute pairwise squared Euclidean distances
    dists = cdist(X1_scaled, X2_scaled, metric='sqeuclidean')

    # Apply kernel formula
    return np.exp(-0.5 * dists)

def negative_log_marginal_likelihood(params, X_train, y_train):
    """
    Compute the negative log marginal likelihood (NLML) for a GP with exponential kernel.
    Parameters:
        params: (theta, noise)
    Returns:
        Scalar NLML value
    """
    theta, noise = params
    epsilon = 1e-6

    # Compute kernel matrix with jitter
    K = exponential_kernel(X_train, X_train, theta) + (noise ** 2 + epsilon) * np.eye(len(X_train))

    try:
        # Cholesky decomposition
        L = np.linalg.cholesky(K)
        log_det_K = 2 * np.sum(np.log(np.diag(L)))

        # Solve instead of inverse
        alpha = np.linalg.solve(K, y_train)
        nlml = 0.5 * y_train.T @ alpha + 0.5 * log_det_K + 0.5 * len(y_train) * np.log(2 * np.pi)

        return float(nlml)

    except np.linalg.LinAlgError:
        print("Cholesky failed — ill-conditioned matrix")
        return np.inf

def optimize_hyperparameters(X_train, y_train):
    """
    Optimize the Gaussian Process hyperparameters (constant mean, length scale, sigma_f, noise).
    
    Parameters:
        X_train: Training inputs (NxD)
        y_train: Training targets (Nx1)

    Returns:
        Optimized hyperparameters: (mean, length_scale, sigma_f, noise)
    """
    # Initial guesses: theta, noise
    initial_params = [1e-2, 1e-2]  # lengthscale, noise

    
    # Bounds to ensure positivity of kernel hyperparameters
    bounds = [(1e-2, 30), (1e-3, 10)]

    # Minimize the negative log marginal likelihood
    res = minimize(negative_log_marginal_likelihood, 
                   initial_params, args=(X_train, y_train),
                   method='L-BFGS-B', bounds=bounds)

    return res.x  # Return optimized hyperparameters (theta, noise)

def gp_regression(X_train, y_train, X_test, theta, noise):
    """
    Compute the posterior mean and covariance for Gaussian Process Regression (GPR).

    Parameters:
        X_train: Training inputs (NxD)
        y_train: Training targets (Nx1)
        X_test: Test inputs (MxD)
        mean: Constant mean parameter
        length_scale: Kernel length-scale
        sigma_f: Kernel variance
        noise: Observation noise standard deviation

    Returns:
        mu_post: Posterior mean at X_test (Mx1)
        Sigma_post: Posterior covariance matrix (MxM)
    """
    epsilon = 1e-6  # Jitter term to stabilize inversion

    # Compute kernel matrices with jitter
    K = exponential_kernel(X_train, X_train, theta) + (noise**2 + epsilon) * np.eye(len(X_train)) #(training covariance)
    K_s = exponential_kernel(X_train, X_test, theta) # (cross covariance)
    K_ss = exponential_kernel(X_test, X_test, theta) + (noise**2 + epsilon) * np.eye(len(X_test)) #(test covariance)

    L = np.linalg.cholesky(K)  # Compute Cholesky factorization
    K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(K.shape[0])))  # Compute K_inv efficiently

    # Compute posterior mean
    mu_post =  K_s.T @ K_inv @ y_train

    # Compute posterior covariance
    Sigma_post = K_ss - K_s.T @ K_inv @ K_s
    
    # Ensure symmetry
    Sigma_post = (Sigma_post + Sigma_post.T) / 2  
    
    # Eigenvalue correction (shift negative values)
    eigvals, eigvecs = np.linalg.eigh(Sigma_post)
    eigvals[eigvals < 1e-5] = 1e-5  # Shift negative eigenvalues to a small positive value
    Sigma_post = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Ensure all values are non-negative
    Sigma_post = np.maximum(Sigma_post, 0)
    return mu_post, Sigma_post

def negative_log_marginal_likelihood_weighted(params, X_train, y_train, w_D):
    
    theta, noise = params

    # Compute kernel matrices
    K = exponential_kernel(X_train, X_train, theta)

    # Clip w_D to avoid extreme small values
    w_D = np.clip(w_D, 1e-3, 1)

    # Construct weighted noise covariance matrix
    W = noise**2 * np.diag(1.0 / w_D)  # W = sigma^2 diag(1/w_i)

    # Add jitter for numerical stability
    jitter = 1e-5 * np.eye(len(X_train))
    K_W = K + W + jitter

    # Use Cholesky decomposition instead of direct inversion
    try:
        L = np.linalg.cholesky(K_W)
        K_W_inv = scipy.linalg.cho_solve((L, True), np.eye(len(X_train)))  # Stable inverse
    except np.linalg.LinAlgError:
        print("Warning: Singular matrix detected! Increasing jitter...")
        jitter *= 10  # Increase jitter and retry
        K_W = K + W + jitter
        L = np.linalg.cholesky(K_W)
        K_W_inv = scipy.linalg.cho_solve((L, True), np.eye(len(X_train)))

    # Compute log determinant safely
    logdet_K_W = 2 * np.sum(np.log(np.diag(L)))  # log(det(K_W)) from Cholesky


    # Compute NLML
    nlml = 0.5 * (y_train.T @ K_W_inv @ y_train) + 0.5 * logdet_K_W + 0.5 * len(X_train) * np.log(2 * np.pi)

    return nlml.item()  # Return scalar value


def optimize_hyperparameters_weighted(X_train, y_train, w_D):
    """
    Optimize the Gaussian Process hyperparameters (constant mean, length scale, sigma_f, noise)
    for the Weighted Gaussian Process model.

    Parameters:
        X_train: Training inputs (NxD)
        y_train: Training targets (Nx1)
        w_D: Weight vector (N,) representing the weighting of each observation.

    Returns:
        Optimized hyperparameters: (mean, length_scale, sigma_f, noise)
    """
    # Initial guesses: theta, noise
    initial_params = [1e-4,  0.1]  

    # Bounds to ensure positivity of kernel hyperparameters
    bounds = [(1e-2, 30), (1e-3, 10)]

    # Minimize the negative log marginal likelihood
    res = minimize(negative_log_marginal_likelihood_weighted,
                   initial_params, 
                   args=(X_train, y_train, w_D),
                   method='L-BFGS-B', 
                   bounds=bounds)

    return res.x  # Return optimized hyperparameters (mean, theta, noise)


def gp_regression_weighted(X_train, y_train, X_test, theta, noise, w_D):
    
    # Compute standard kernel matrices
    K = exponential_kernel(X_train, X_train, theta)
    K_s = exponential_kernel(X_train, X_test, theta)
    K_ss = exponential_kernel(X_test, X_test, theta)

    # Ensure weights are valid
    w_D = np.clip(w_D, 1e-3, 1)  # Avoid division by zero & extreme values

    # Compute the weight matrix W (diagonal matrix)
    W = noise**2 * np.diag(1.0 / w_D)  # W = sigma^2 diag(1/w_i)

    # Small jitter term for numerical stability
    jitter = 1e-5 * np.eye(len(X_train))

    # Compute the weighted covariance matrix
    K_W = K + W + jitter  # Adding jitter

    # **Use Cholesky decomposition instead of direct inversion**
    try:
        L = np.linalg.cholesky(K_W)  # More stable than np.linalg.inv()
        K_W_inv = scipy.linalg.cho_solve((L, True), np.eye(len(X_train)))
    except np.linalg.LinAlgError:
        print("Warning: Matrix was singular, adding more jitter.")
        jitter *= 10  # Increase jitter
        L = np.linalg.cholesky(K_W + jitter)
        K_W_inv = scipy.linalg.cho_solve((L, True), np.eye(len(X_train)))

    # Compute posterior mean
    
    mu_post =  K_s.T @ K_W_inv @ y_train

    # Compute posterior covariance
    Sigma_post = K_ss - K_s.T @ K_W_inv @ K_s

    # Ensure symmetry
    Sigma_post = (Sigma_post + Sigma_post.T) / 2  
    
    # Eigenvalue correction (shift negative values)
    eigvals, eigvecs = np.linalg.eigh(Sigma_post)
    eigvals[eigvals < 1e-5] = 1e-5  # Shift negative eigenvalues to a small positive value
    Sigma_post = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Ensure all values are non-negative
    Sigma_post = np.maximum(Sigma_post, 0)
    
    return mu_post, Sigma_post




def negative_loglikelihood_weight(params, X_train, y_train, depth):
    gamma, lamda = params
    weights = weight(1-depth, gamma, lamda)
    parameters = optimize_hyperparameters_weighted(X_train, y_train, weights)
    nll = negative_log_marginal_likelihood_weighted(parameters, X_train, y_train, weights)
    return nll

# Optimizing gamma and lamda for DWGP
def optimize_weight_parameters(X_train, y_train, depth):
    initial_params = [0.005, 1]
    bounds = [(0, 1), (1,None)]
    res = minimize(negative_loglikelihood_weight, initial_params, args = (X_train, y_train, depth), method='L-BFGS-B', bounds=bounds)
    return res.x

# Negative loglikelihood for Optimizing gamma for RWGP
def nll_RWGP(parameter, X_train, y_train, r, s):
    gamma = parameter
    w_D = compute_robust_weights(X_train, y_train,  r, s, gamma, eta=0.5)
    mean, length_scale, sigma_f, noise = optimize_hyperparameters_weighted(X_train, y_train, w_D)
    params = (mean, length_scale, sigma_f, noise)
    nll = negative_log_marginal_likelihood_weighted(params, X_train, y_train, w_D)
    return nll
    
# Optimizing gamma for RWGP    
def optimize_gamma(X_train, y_train, r, s):
    initial_gamma = 0.005
    bound = [(0,1)]
    res = minimize(nll_RWGP, initial_gamma, args = (X_train, y_train, r, s), method = 'powell', bounds = bound)
    return res.x    
    