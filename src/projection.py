import numpy as np
from numpy.testing import assert_almost_equal
import typing

def l0_norm(a: np.array):
    return np.count_nonzero(a)


def simplex_proj(a: np.array):

    assert (a >= 0).all()

    d = len(a)

    if np.sum(a) <= 1 + 1e-6:
        return a, d, 0.0

    # sort largest to smallest
    a_ = np.flip(np.sort(a))

    # find d_0
    a_cs = np.cumsum(a_) - 1
    criterion = a_ > (a_cs / np.arange(1, d + 1))
    d_0 = np.sum(criterion)

    theta = a_cs[d_0 - 1] / d_0

    return np.maximum(0.0, a - theta), d_0, theta


def l1_ball_proj(a: np.array, radius: float):
    assert radius > 0
    d = len(a)

    abs_a = np.abs(a)
    if np.sum(abs_a) <= radius + 1e-6:
        return a, d, 0.0
    else:
        sign = np.sign(a)
        simplex_x, d_0, theta = simplex_proj(abs_a / radius)
        proj = radius * sign * simplex_x
        assert (np.sum(np.abs(proj)) - radius) < 1e-5
        return proj, d_0, theta


if __name__ == "__main__":

    # norm_0(0) == 0
    x = np.zeros(3)
    norm = l0_norm(x)
    assert norm == 0.0

    # norm_0(1) == len(1)
    x = np.ones(3)
    norm = l0_norm(x)
    assert norm == len(x)

    # norm_0(rand(3)) == 3 (a.s.)
    x = np.random.rand(3)
    norm = l0_norm(x)
    assert norm == len(x)

    # norm_0(1e-5) == len(1)
    x = 1e-12 * np.ones(3)
    norm = l0_norm(x)
    assert norm == len(x)

    x = np.array([1.1, 0.6])
    x_proj, d_0, theta = simplex_proj(x)
    assert d_0 == l0_norm(x_proj)

    # exterior point projects to frontier
    x = np.array([0.6, 1.1])
    x_proj, d_0, theta = simplex_proj(x)
    assert_almost_equal(np.sum(x_proj), 1.0, decimal=8)
    assert d_0 == l0_norm(x_proj)

    # assert
    x = np.array([0.0, 1.1])
    x_proj, d_0, theta = simplex_proj(x)
    assert_almost_equal(np.array([0.0, 1.0]), x_proj, decimal=8)
    assert d_0 == l0_norm(x_proj)

    x = np.array([1.1, 0.0])
    x_proj, d_0, theta = simplex_proj(x)
    assert_almost_equal(np.array([1.0, 0.0]), x_proj, decimal=8)
    assert d_0 == l0_norm(x_proj)

    # assert inside simplex identical
    for i in range(100):
        x = np.random.rand(100)
        x = x / np.sum(x)
        x_proj, d_0, theta = simplex_proj(x)
        assert_almost_equal(x, x_proj, decimal=8)
        assert d_0 == l0_norm(x_proj)

    # assert inside B(z) identical
    for i in range(100):
        radius = 10
        x = np.random.rand(100)
        x = x / np.sum(x) * radius
        x_proj, d_0, theta = l1_ball_proj(x, radius)
        assert_almost_equal(x, x_proj, decimal=8)
        assert 100 == l0_norm(x_proj)

    # assert
    x = np.array([0.0, 1.1])
    x_proj, d_0, theta = l1_ball_proj(x, 1)
    assert_almost_equal(np.array([0.0, 1.0]), x_proj, decimal=8)
    assert d_0 == l0_norm(x_proj)

    x = np.array([1.1, 0.0])
    x_proj, d_0, theta = l1_ball_proj(x, 1)
    assert_almost_equal(np.array([1.0, 0.0]), x_proj, decimal=8)
    assert d_0 == l0_norm(x_proj)

    x = np.array([1.1, 0.1])
    x_proj, d_0, theta = l1_ball_proj(x, 1)
    assert_almost_equal(np.array([1.0, 0.0]), x_proj, decimal=8)
    assert d_0 == l0_norm(x_proj)

    x = np.array([10.10, 0.0])
    x_proj, d_0, theta = l1_ball_proj(x, 10)
    assert_almost_equal(np.array([10.0, 0.0]), x_proj, decimal=8)
    assert d_0 == l0_norm(x_proj)

    x = np.array([10.10, 1.0, 45.0])
    x_proj, d_0, theta = l1_ball_proj(x, 10)
    assert (np.sum(np.abs(x_proj)) - radius) < 1e-5
