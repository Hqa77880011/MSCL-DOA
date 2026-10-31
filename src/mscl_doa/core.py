import numpy as np
import cvxpy as cp
from scipy.signal import find_peaks
from scipy.optimize import nnls, linear_sum_assignment


def steering(positions, angles):
    return np.exp(
        2j * np.pi * np.asarray(positions)[:, None] * np.sin(np.deg2rad(angles))[None]
    )


def observations(x, masks, c):
    records = []
    for block, mask in zip(x, masks):
        ids = np.flatnonzero(mask)
        values = block[ids]
        if not len(ids):
            continue
        covariance = values @ values.conj().T / values.shape[1]
        shrunk = (1 - c["shrinkage"]) * covariance + c["shrinkage"] * np.trace(
            covariance
        ).real / len(ids) * np.eye(len(ids))
        omega = np.kron(shrunk, shrunk.conj())
        omega += (
            1e-8
            * max(float(np.trace(omega).real / len(omega)), 1e-8)
            * np.eye(len(omega))
        )
        records.append((ids, covariance, omega, values.shape[1]))
    return records


def music(covariance, positions, sources, step):
    values, vectors = np.linalg.eigh((covariance + covariance.conj().T) / 2)
    if sources >= len(positions):
        raise ValueError("Source count must be smaller than sensor count")
    noise = vectors[:, : len(positions) - sources]
    grid = np.arange(-80, 80 + step / 2, step)
    a = steering(positions, grid)
    spectrum = 1 / np.maximum(np.sum(np.abs(noise.conj().T @ a) ** 2, axis=0), 1e-12)
    peaks = find_peaks(spectrum, distance=max(1, int(1 / step)))[0]
    if len(peaks) < sources:
        peaks = np.arange(len(grid))
    chosen = peaks[np.argsort(spectrum[peaks])[-sources:]]
    return np.sort(grid[chosen]), grid, spectrum


def fit(x, masks, positions, c, method="mscl"):
    positions = np.asarray(positions, float)
    lag_matrix = np.round(positions[:, None] - positions[None, :], 8)
    lags = np.unique(lag_matrix)
    index = np.searchsorted(lags, lag_matrix)
    records = observations(x, masks, c)
    if not records:
        raise ValueError("No active sensors")
    count = len(lags)
    information = np.zeros((count, count), complex)
    rhs = np.zeros(count, complex)
    terms = []
    variable = cp.Variable(count, complex=True)
    for ids, R, omega, snapshots in records:
        observed = index[np.ix_(ids, ids)].ravel()
        design = np.eye(count)[observed]
        if method in ("pair_count", "completion"):
            precision = np.eye(len(observed))
        elif method == "diagonal":
            precision = np.diag(1 / np.maximum(np.diag(omega).real, 1e-12))
        else:
            precision = np.linalg.inv(omega)
        values, vec = np.linalg.eigh(precision)
        root = np.diag(np.sqrt(np.maximum(values, 0))) @ vec.conj().T
        terms.append(
            snapshots * cp.sum_squares(cp.abs(root @ (design @ variable - R.ravel())))
        )
        information += snapshots * design.conj().T @ precision @ design
        rhs += snapshots * design.conj().T @ precision @ R.ravel()
    physical = cp.bmat(
        [
            [
                cp.reshape(variable[index[i, j]], (1, 1), order="C")
                for j in range(len(positions))
            ]
            for i in range(len(positions))
        ]
    )
    constraints = [physical == physical.H, physical >> 0]
    constraints += [
        variable[k] == cp.conj(variable[np.searchsorted(lags, -lag)])
        for k, lag in enumerate(lags)
    ]
    regularizer = cp.sum_squares(cp.abs(variable))
    penalty = c["regularization"] * (10 if method == "completion" else 1)
    problem = cp.Problem(cp.Minimize(sum(terms) + penalty * regularizer), constraints)
    problem.solve(solver=c["solver"])
    if (
        problem.status not in ("optimal", "optimal_inaccurate")
        or variable.value is None
    ):
        raise RuntimeError(f"Covariance solve failed: {problem.status}")
    lag_state = variable.value
    covariance = lag_state[index]
    angles, grid, spectrum = music(covariance, positions, c["sources"], c["grid_step"])
    quality, rank = reliability(lag_state, lags, information, angles)
    return dict(
        angles=angles,
        quality=quality,
        information_rank=rank,
        lag_coverage=np.count_nonzero(np.diag(information).real > 1e-10) / count,
        grid=grid,
        spectrum=spectrum,
        covariance=covariance,
    )


def reliability(lag_state, lags, information, angles):
    phase = np.exp(2j * np.pi * lags[:, None] * np.sin(np.deg2rad(angles))[None])
    nuisance = np.column_stack([phase, (np.abs(lags) < 1e-9).astype(float)])
    powers = nnls(
        np.r_[nuisance.real, nuisance.imag], np.r_[lag_state.real, lag_state.imag]
    )[0][:-1]
    derivative = (
        phase
        * (2j * np.pi * lags[:, None] * np.cos(np.deg2rad(angles))[None])
        * powers[None]
        * np.pi
        / 180
    )
    jac = np.column_stack([derivative, nuisance])
    fisher = 2 * np.real(jac.conj().T @ information @ jac)
    k = len(angles)
    profiled = (
        fisher[:k, :k]
        - fisher[:k, k:] @ np.linalg.pinv(fisher[k:, k:], rcond=1e-9) @ fisher[k:, :k]
    )
    eigen = np.linalg.eigvalsh((profiled + profiled.T) / 2)
    positive = eigen[eigen > max(eigen.max(initial=0) * 1e-9, 1e-12)]
    quality = (
        float(positive.min() / (np.trace(profiled) / k + 1e-12))
        if len(positive)
        else 0.0
    )
    return quality, len(positive)


def matched_error(prediction, truth):
    errors = np.abs(np.asarray(prediction)[:, None] - np.asarray(truth)[None])
    left, right = linear_sum_assignment(errors)
    return errors[left, right]
