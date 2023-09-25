__all__ = [
    "sigma",
    "inv_sigma",
    "sech",
    "relu",
    "RK_solver",
    "cum_integral",
    "integral",
    "gaussian",
    "target_traj",
    "_aver_kernel",
    "_aver",
]

import numpy as np
from numpy import zeros, empty
import numba
from numba import njit
from elastica._linalg import _batch_matvec, _batch_cross


def sigma(V, V_rest=0.0, V_ub=80.0, gap=0.01):
    mean = 0.5 * (V_rest + V_ub)
    var = 2 / (V_rest + V_ub) * np.arctanh(2 * gap - 1)
    if mean > 0:
        var *= -1
    array = 0.5 + 0.5 * np.tanh(var * (V - mean))
    return array


def inv_sigma(u, V_rest=0.0, V_ub=80.0, gap=0.01):
    mean = 0.5 * (V_rest + V_ub)
    var = 2 / (V_rest + V_ub) * np.arctanh(2 * gap - 1)
    if mean > 0:
        var *= -1
    u = np.clip(u, gap, 1 - gap)
    array = mean + 1 / var * np.arctanh(2 * u - 1)
    return array - V_rest


def sech(x):
    return 2 * np.exp(x) / (np.exp(2 * x) + 1)


def relu(x):
    try:
        return max(0.0, x)
    except:
        array = x.copy()
        array[x < 0] *= 0
        return array


def RK_solver(dynamics, V, dt, I):
    V_current = V.copy()
    k1 = dynamics(V_current, I)
    k2 = dynamics(V_current + k1 * dt / 2, I)
    k3 = dynamics(V_current + k2 * dt / 2, I)
    k4 = dynamics(V_current + k3 * dt, I)
    dVdt = 1 / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    V = V_current + dVdt * dt
    return V


def cum_integral(vector, ds):
    array = np.cumsum(vector, axis=-1) * ds
    return array


def integral(vector, ds):
    array = np.sum(vector, axis=-1) * ds
    return array


def gaussian(x, mu, sigma, magnitude):
    return np.exp(-((x - mu) ** 2.0) / (2 * sigma**2.0)) * magnitude


## target position
def target_traj(target, start_pos, end_pos, start_time, end_time, total_steps):
    start_idx = int(total_steps * start_time)
    end_idx = int(total_steps * end_time)
    time_step = end_idx - start_idx
    target[start_idx:end_idx, :] = np.vstack(
        [
            start_pos[0] + np.linspace(0, 1, time_step) * (end_pos[0] - start_pos[0]),
            start_pos[1] + np.linspace(0, 1, time_step) * (end_pos[1] - start_pos[1]),
        ]
    ).T


@njit(cache=True)
def _row_sum(array_collection):
    rowsize = array_collection.shape[0]
    array_sum = np.zeros(array_collection.shape[1:])
    for n in range(rowsize):
        array_sum += array_collection[n, ...]
    return array_sum


@njit(cache=True)
def _material_to_lab(director_collection, vectors):
    blocksize = vectors.shape[1]
    lab_frame_vectors = np.zeros((3, blocksize))
    for n in range(blocksize):
        for i in range(3):
            for j in range(3):
                lab_frame_vectors[i, n] += director_collection[j, i, n] * vectors[j, n]
    return lab_frame_vectors


@njit(cache=True)
def _lab_to_material(director_collection, vectors):
    return _batch_matvec(director_collection, vectors)


@njit(cache=True)
def _aver_kernel(array_collection):
    """
    Simple trapezoidal quadrature rule with zero at end-points, in a dimension agnostic way

    Parameters
    ----------
    array_collection

    Returns
    -------
    Notes
    -----
    Micro benchmark results, for a block size of 100, using timeit
    Python version: 8.14 µs ± 1.42 µs per loop
    This version: 781 ns ± 18.3 ns per loop
    """
    blocksize = array_collection.shape[-1]
    temp_collection = empty(array_collection.shape[:-1] + (blocksize + 1,))

    temp_collection[..., 0] = 0.5 * array_collection[..., 0]
    temp_collection[..., blocksize] = 0.5 * array_collection[..., blocksize - 1]

    for k in range(1, blocksize):
        temp_collection[..., k] = 0.5 * (
            array_collection[..., k] + array_collection[..., k - 1]
        )
    return temp_collection


@njit(cache=True)
def _diff_kernel(array_collection):
    """
    This function does differentiation.
    Parameters
    ----------
    array_collection

    Returns
    -------
    Notes
    -----
    Micro benchmark results showed that for a block size of 100, using timeit
    Python version: 9.07 µs ± 2.15 µs per loop
    This version: 952 ns ± 91.1 ns per loop
    """
    blocksize = array_collection.shape[-1]
    temp_collection = empty(array_collection.shape[:-1] + (blocksize + 1,))

    temp_collection[..., 0] = array_collection[..., 0]
    temp_collection[..., blocksize] = -array_collection[..., blocksize - 1]

    for k in range(1, blocksize):
        temp_collection[..., k] = (
            array_collection[..., k] - array_collection[..., k - 1]
        )
    return temp_collection


@njit(cache=True)
def _diff(array_collection):
    """
    This function computes difference between elements of a batch vector
    Parameters
    ----------
    vector

    Returns
    -------
    Notes
    -----
    Micro benchmark results showed that for a block size of 100, using timeit
    Python version: 3.29 µs ± 767 ns per loop
    This version: 840 ns ± 14.5 ns per loop
    """
    blocksize = array_collection.shape[-1]
    output_vector = empty(array_collection.shape[:-1] + (blocksize - 1,))

    for k in range(1, blocksize):
        output_vector[..., k - 1] = (
            array_collection[..., k] - array_collection[..., k - 1]
        )
    return output_vector


@njit(cache=True)
def _aver(array_collection):
    """
    This function computes the average between elements of a vector
    Parameters
    ----------
    vector

    Returns
    -------
    Notes
    -----
    Micro benchmark results showed that for a block size of 100, using timeit
    Python version: 2.37 µs ± 764 ns per loop
    This version: 713 ns ± 3.69 ns per loop
    """
    blocksize = array_collection.shape[-1]
    output_vector = empty(array_collection.shape[:-1] + (blocksize - 1,))

    for k in range(1, blocksize):
        output_vector[..., k - 1] = 0.5 * (
            array_collection[..., k] + array_collection[..., k - 1]
        )
    return output_vector
