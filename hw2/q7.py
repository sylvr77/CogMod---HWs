import numpy as np
from scipy.stats import multivariate_normal


# 1) Function: multivariate_normal_density(x, mu, Sigma)
def multivariate_normal_logpdf(x, mu, Sigma):
    """
    Vectorized log-pdf of N(mu, Sigma) at x.

    Parameters
    ----------
    x : array_like, shape (..., D) or (D,)
    mu : array_like, shape (D,)
    Sigma : array_like, shape (D, D)

    Returns
    -------
    logp : ndarray, shape (...)  (scalar if x is (D,))
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    if mu.ndim != 1:
        raise ValueError("mu must be a 1D array of shape (D,).")
    D = mu.shape[0]
    if Sigma.shape != (D, D):
        raise ValueError(f"Sigma must have shape ({D},{D}).")
    if x.shape == (D,):
        x = x[None, :]  # make it (1, D) for unified vectorization
        squeeze_out = True
    else:
        if x.shape[-1] != D:
            raise ValueError(f"x must have last dimension D={D}. Got {x.shape[-1]}.")
        squeeze_out = False

    # Cholesky: Sigma = L L^T (requires SPD covariance)
    L = np.linalg.cholesky(Sigma)

    # Compute quadratic form (x-mu)^T Sigma^{-1} (x-mu) efficiently:
    # Solve L y = (x-mu)^T  -> y = L^{-1}(x-mu)^T
    xc = x - mu  # shape (N, D) (where N is product of leading dims)
    # reshape leading dims to a flat batch for solving
    orig_shape = xc.shape[:-1]
    xc2 = xc.reshape(-1, D)  # (Nflat, D)
    y = np.linalg.solve(L, xc2.T)  # (D, Nflat)
    quad = np.sum(y * y, axis=0)   # (Nflat,)

    # log det(Sigma) = 2 * sum(log(diag(L)))
    logdet = 2.0 * np.sum(np.log(np.diag(L)))

    log2pi = np.log(2.0 * np.pi)
    logp = -0.5 * (D * log2pi + logdet + quad)
    logp = logp.reshape(orig_shape)

    if squeeze_out:
        return logp[0]
    return logp


def multivariate_normal_density(x, mu, Sigma):
    """pdf = exp(logpdf), vectorized like multivariate_normal_logpdf."""
    return np.exp(multivariate_normal_logpdf(x, mu, Sigma))


# 2) Class with SciPy-like semantics: rvs(size=...) and log_pdf(x)
class MultivariateNormal:
    """
    Minimal SciPy-like multivariate normal with:
      - rvs(size=None, random_state=None)
      - log_pdf(x)  (vectorized)

    Notes:
      * size can be None, int, or tuple
      * samples have shape: size + (D,)
      * x can be shape (..., D) or (D,)
    """
    def __init__(self, mu, Sigma):
        self.mu = np.asarray(mu, dtype=float)
        self.Sigma = np.asarray(Sigma, dtype=float)

        if self.mu.ndim != 1:
            raise ValueError("mu must be 1D (D,).")
        self.D = self.mu.shape[0]
        if self.Sigma.shape != (self.D, self.D):
            raise ValueError(f"Sigma must be ({self.D},{self.D}).")

        # Precompute Cholesky + logdet for speed in repeated calls
        self.L = np.linalg.cholesky(self.Sigma)
        self.logdet = 2.0 * np.sum(np.log(np.diag(self.L)))
        self._log_norm_const = -0.5 * (self.D * np.log(2.0 * np.pi) + self.logdet)

    def log_pdf(self, x):
        x = np.asarray(x, dtype=float)

        if x.shape == (self.D,):
            x = x[None, :]
            squeeze_out = True
        else:
            if x.shape[-1] != self.D:
                raise ValueError(f"x must have last dimension {self.D}. Got {x.shape[-1]}.")
            squeeze_out = False

        xc = x - self.mu
        orig_shape = xc.shape[:-1]
        xc2 = xc.reshape(-1, self.D)              # (Nflat, D)
        y = np.linalg.solve(self.L, xc2.T)        # (D, Nflat)
        quad = np.sum(y * y, axis=0)              # (Nflat,)
        logp = self._log_norm_const - 0.5 * quad
        logp = logp.reshape(orig_shape)

        if squeeze_out:
            return logp[0]
        return logp

    def rvs(self, size=None, random_state=None):
        rng = np.random.default_rng(random_state)

        if size is None:
            z = rng.standard_normal(self.D)  # (D,)
            return self.mu + self.L @ z

        if isinstance(size, int):
            size = (size,)
        else:
            size = tuple(size)

        z = rng.standard_normal(size + (self.D,))   # (..., D)
        # (L @ z) over last axis: use einsum for clear batch multiplication
        samples = self.mu + np.einsum("ij,...j->...i", self.L, z)
        return samples
    

# 3) Compare against SciPy for spherical, diagonal, and full covariance
def check_against_scipy():
    rng = np.random.default_rng(0)

    # Dimensions and test points
    D = 4
    x = rng.normal(size=(10, D))
    mu = rng.normal(size=D)

    # 1) Spherical: Sigma = s^2 I
    s2 = 1.7
    Sigma_sph = s2 * np.eye(D)

    # 2) Diagonal: different variances
    diag = np.array([0.5, 1.2, 2.0, 3.3])
    Sigma_diag = np.diag(diag)

    # 3) Full covariance: construct SPD matrix
    A = rng.normal(size=(D, D))
    Sigma_full = A @ A.T + 0.1 * np.eye(D)  # SPD

    cases = [
        ("spherical", Sigma_sph),
        ("diagonal",  Sigma_diag),
        ("full",      Sigma_full),
    ]

    for name, Sigma in cases:
        my_pdf = multivariate_normal_density(x, mu, Sigma)
        sp_pdf = multivariate_normal(mean=mu, cov=Sigma).pdf(x)

        my_log = multivariate_normal_logpdf(x, mu, Sigma)
        sp_log = multivariate_normal(mean=mu, cov=Sigma).logpdf(x)

        print(f"\nCase: {name}")
        print("  max |pdf diff|   =", np.max(np.abs(my_pdf - sp_pdf)))
        print("  max |log diff|   =", np.max(np.abs(my_log - sp_log)))
        print("  mean |log diff|  =", np.mean(np.abs(my_log - sp_log)))

    # Class check
    Sigma = Sigma_full
    dist = MultivariateNormal(mu, Sigma)
    sp = multivariate_normal(mean=mu, cov=Sigma)

    x2 = rng.normal(size=(3, 5, D))  # extra batch dims
    print("\nClass log_pdf vs SciPy logpdf, max diff:",
          np.max(np.abs(dist.log_pdf(x2) - sp.logpdf(x2))))

    samples = dist.rvs(size=(10000,), random_state=0)
    print("Sample mean error (norm):", np.linalg.norm(samples.mean(axis=0) - mu))
    print("Sample cov error (Fro):  ", np.linalg.norm(np.cov(samples, rowvar=False) - Sigma, ord="fro"))

check_against_scipy()
