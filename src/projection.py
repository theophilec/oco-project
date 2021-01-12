import typing

import numpy as np
from numpy.testing import assert_almost_equal


def l0_norm(x: np.array):
    """Compute the L_0 norm of x."""
    return np.count_nonzero(x)


def simplex_proj(x: np.array):
    """Project x onto the unit simplex."""

    assert (x >= 0).all()

    d = len(x)

    if np.sum(x) <= 1 + 1e-6:
        return x, d, 0.0

    # sort largest to smallest
    x_ = np.flip(np.sort(x))

    # find d_0
    x_cs = np.cumsum(x_) - 1
    criterion = x_ > (x_cs / np.arange(1, d + 1))
    d_0 = np.sum(criterion)

    theta = x_cs[d_0 - 1] / d_0

    return np.maximum(0.0, x - theta), d_0, theta


def l1_ball_proj(x: np.array, radius: float):
    """Project x onto L1 ball according to l2 norm."""
    assert radius > 0
    d = len(x)

    abs_x = np.abs(x)
    if np.sum(abs_x) <= radius + 1e-6:
        return x, d, 0.0
    else:
        sign = np.sign(x)
        simplex_x, d_0, theta = simplex_proj(abs_x / radius)
        proj = radius * sign * simplex_x
        assert (np.sum(np.abs(proj)) - radius) < 1e-5
        return proj, d_0, theta


def l1_ball_proj_weighted(x: np.array, radius: float, D: np.array):
    """Project x onto L1 ball according to weighted l2 norm."""
    assert radius > 0
    d = len(x)

    if np.sum(np.abs(x)) <= radius + 1e-6:
        return x, d, 0.0
    else:
        x_abs = np.abs(x) / radius

        Dx = D * x_abs
        argsort = np.argsort(Dx)[::-1]
        assert Dx[argsort][0] >= Dx[argsort][1]

        # find d_0 and theta
        x_cs = np.cumsum(x_abs[argsort])
        Dinv_cs = np.cumsum(1 / D[argsort])
        criterion = Dx[argsort] >= ((x_cs - 1) / Dinv_cs)
        d_0 = np.sum(criterion)

        theta = (x_cs[d_0 - 1] - 1) / Dinv_cs[d_0 - 1]
        # projection of abs(x) / radius
        simplex_proj = 1 / D * np.maximum(0.0, Dx - theta)
        proj = simplex_proj * np.sign(x) * radius
        assert (np.sum(np.abs(proj)) - radius) < 1e-5
        return proj, d_0, theta

if __name__ == "__test__":
    d = 10
    x = np.random.randn(d) * 20
    radius = 5
    D = np.ones(d)
    y = l1_ball_proj_weighted(x, radius, D)
    y_ = l1_ball_proj(x, radius)
    assert y == y_

