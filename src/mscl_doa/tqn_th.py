"""Wu et al. (2025) TQN-TH, following the published Eqs. (16)--(34)/Algorithm 1.

DOI: 10.1007/s11760-025-04544-3. This implements the printed updates,
including Eq. (22)'s zero-filled observation matrix. It does not substitute
a nuclear/trace penalty or a newly derived masked proximal algorithm.
"""
import numpy as np


def toeplitz_hermitian(matrix):
    """Orthogonal Toeplitz/Hermitian projection by paired diagonal means (28)."""
    matrix = np.asarray(matrix, complex)
    n = len(matrix)
    first = np.empty(n, complex)
    first[0] = np.diag(matrix).real.mean()
    for lag in range(1, n):
        first[lag] = (np.diag(matrix, -lag).mean() + np.diag(matrix, lag).mean().conjugate()) / 2
    difference = np.subtract.outer(np.arange(n), np.arange(n))
    return np.where(difference >= 0, first[abs(difference)], first[abs(difference)].conjugate())


def positive_projection(matrix, floor):
    """Hermitian eigendecomposition and the eigenvalue floor of Eq. (30)."""
    eigenvalues, vectors = np.linalg.eigh((matrix + matrix.conj().T) / 2)
    return (vectors * np.maximum(eigenvalues, floor)) @ vectors.conj().T


def tqn_svt(matrix, alpha, mu, rho):
    """Published TQN-SVT: keep large singular values and scale small ones (24)."""
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    threshold = np.sqrt(1 / alpha)
    shrunk = np.where(singular >= threshold, singular, singular / (1 + 2 * alpha / (mu + rho)))
    return (left * shrunk) @ right


def initial_toeplitz(observed, mask):
    """Algorithm 1 initialization: average known entries at each physical lag."""
    n = len(observed)
    differences = np.subtract.outer(np.arange(n), np.arange(n))
    initial = np.zeros((n, n), complex)
    for lag in range(1 - n, n):
        selected = (differences == lag) & mask
        if selected.any():
            initial[differences == lag] = observed[selected].mean()
    return initial


def complete(observed, mask, c):
    """Complete a virtual covariance with the published TQN-TH ADMM iterations.

    The definition's tau=sqrt(1/alpha) is used despite Section 4's conflicting
    tau=1/alpha sentence. Eq. (22) is applied literally; no masked correction
    or claim that it is an exact minimizer of Eq. (21) is added here.
    """
    observed = np.asarray(observed, complex)
    mask = np.asarray(mask, bool)
    if observed.ndim != 2 or observed.shape[0] != observed.shape[1] or mask.shape != observed.shape:
        raise ValueError("TQN-TH needs equally sized square observation/mask matrices")
    if not mask.any() or not np.isfinite(observed[mask]).all():
        raise ValueError("TQN-TH needs finite observed covariance entries")
    alpha = float(c.get("tqn_alpha", .03))
    mu = float(c.get("tqn_mu", .01))
    rho = float(c.get("tqn_rho", 3.))
    tolerance = float(c.get("tqn_tolerance", 1e-4))
    projection_tolerance = float(c.get("tqn_projection_tolerance", 1e-6))
    numerical_epsilon = float(c.get("tqn_numerical_epsilon", 1e-8))
    floor = float(c.get("tqn_psd_floor", tolerance))
    maximum = int(c.get("tqn_max_iterations", 200))
    projection_maximum = int(c.get("tqn_projection_iterations", 10))
    if min(alpha, mu, rho, tolerance, projection_tolerance, numerical_epsilon) <= 0 or floor < 0:
        raise ValueError("TQN parameters/tolerances must be positive and the PSD floor nonnegative")
    if min(maximum, projection_maximum) < 1:
        raise ValueError("TQN iteration limits must be positive")
    r_hat = np.where(mask, observed, 0.)
    z = initial_toeplitz(r_hat, mask)
    multiplier = np.zeros_like(z)
    history = []
    for iteration in range(1, maximum + 1):
        intermediate = z - multiplier / rho + mu / (mu + rho) * (r_hat - z)
        x = tqn_svt(intermediate, alpha, mu, rho)
        z = x + multiplier / rho
        for inner in range(1, projection_maximum + 1):
            previous = z
            z = positive_projection(toeplitz_hermitian(previous), floor)
            projection_change = np.linalg.norm(z - previous) / (np.linalg.norm(previous) + numerical_epsilon)
            if projection_change <= projection_tolerance:
                break
        multiplier = multiplier + rho * (x - z)
        residual = np.linalg.norm(x - z) / (np.linalg.norm(z) + numerical_epsilon)
        history.append(dict(iteration=iteration, primal_residual=float(residual),
                            projection_iterations=inner, projection_change=float(projection_change)))
        if residual <= tolerance:
            break
    eigenvalues = np.linalg.eigvalsh((z + z.conj().T) / 2)
    singular = np.linalg.svd(z, compute_uv=False)
    misfit = np.linalg.norm((z - r_hat)[mask]) ** 2
    diagnostics = dict(
        iterations=len(history), converged=bool(residual <= tolerance), primal_residual=float(residual),
        projection_converged=bool(projection_change <= projection_tolerance),
        toeplitz_residual=float(np.linalg.norm(z - toeplitz_hermitian(z)) / (np.linalg.norm(z) + numerical_epsilon)),
        minimum_eigenvalue=float(eigenvalues.min()), observed_squared_error=float(misfit),
        objective=float(np.minimum(alpha * singular ** 2, 1.).sum() + mu / 2 * misfit),
        alpha=alpha, threshold=float(np.sqrt(1 / alpha)), mu=mu, rho=rho, psd_floor=floor,
    )
    return z, diagnostics, history


def virtual_observation(blocks, masks, positions, spacing=.5, grid_tolerance=1e-6):
    """Pool actually observed pairs onto a virtual ULA; retain each block's count.

    Wu's method assumes one fixed incomplete coarray. For changing masks this
    adapter averages same-lag covariance observations weighted by their snapshot
    counts, and passes the resulting single incomplete matrix to Algorithm 1.
    """
    positions = np.asarray(positions, float)
    if spacing <= 0 or not np.isfinite(positions).all():
        raise ValueError("TQN virtual-grid spacing must be positive and coordinates finite")
    continuous = (positions - positions.min()) / spacing
    lattice = np.rint(continuous).astype(int)
    if np.max(abs(continuous - lattice)) > grid_tolerance:
        raise ValueError("TQN-TH requires sensor coordinates on the declared uniform virtual grid")
    if len(np.unique(lattice)) != len(lattice):
        raise ValueError("TQN-TH needs distinct sensor coordinates")
    n = int(lattice.max()) + 1
    if n < 2:
        raise ValueError("TQN-TH needs a nonzero array aperture")
    sums = np.zeros(2 * n - 1, complex)
    counts = np.zeros(2 * n - 1, float)
    for block, mask in zip(blocks, masks):
        ids = np.flatnonzero(mask)
        if not len(ids):
            continue
        selected = block[ids]
        snapshots = selected.shape[1]
        covariance = selected @ selected.conj().T / snapshots
        lags = lattice[ids, None] - lattice[None, ids]
        np.add.at(sums, lags.ravel() + n - 1, covariance.ravel() * snapshots)
        np.add.at(counts, lags.ravel() + n - 1, snapshots)
    if not np.any(counts):
        raise ValueError("No active sensors")
    lag_values = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    difference = np.subtract.outer(np.arange(n), np.arange(n))
    index = difference + n - 1
    physical_lags = np.unique(lattice[:, None] - lattice[None]) + n - 1
    coverage = np.count_nonzero(counts[physical_lags]) / len(physical_lags)
    return lag_values[index], counts[index] > 0, np.arange(n) * spacing, float(coverage)


def infer(case, c):
    """Virtual coarray completion followed by the repository's common MUSIC stage."""
    from .core import music
    observed, mask, positions, coverage = virtual_observation(
        case["x"], case["masks"], case["positions"], c.get("tqn_grid_spacing", .5),
        c.get("tqn_grid_tolerance", 1e-6),
    )
    covariance, diagnostics, history = complete(observed, mask, c)
    angles, grid, spectrum = music(covariance, positions, c["sources"], c["grid_step"])
    eigenvalues = np.linalg.eigvalsh(covariance)
    k = c["sources"]
    # Report the normalized MUSIC eigengap as the quality score.
    quality = max(float((eigenvalues[-k] - eigenvalues[-k - 1]) / max(eigenvalues[-1], 1e-12)), 0.)
    return dict(angles=angles, quality=quality, information_rank=None, lag_coverage=coverage,
                covariance=covariance, virtual_positions=positions, virtual_sensors=len(positions),
                grid=grid, spectrum=spectrum, diagnostics=diagnostics, history=history)
