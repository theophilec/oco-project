import pytest

import numpy as np
from numpy.testing import assert_almost_equal

from projection import l0_norm, simplex_proj, l1_ball_proj


def test_norm_zeros():
    # norm_0(0) == 0
    x = np.zeros(3)
    norm = l0_norm(x)
    assert norm == 0.0


def test_norm_ones():
    # norm_0(1) == len(1)
    x = np.ones(3)
    norm = l0_norm(x)
    assert norm == len(x)


def test_norm_rand():
    # norm_0(rand(3)) == 3 (a.s.)
    x = np.random.rand(3)
    norm = l0_norm(x)
    assert norm == len(x)


def test_norm_small():
    # norm_0(eps*1) == len(1)
    x = 1e-12 * np.ones(3)
    norm = l0_norm(x)
    assert norm == len(x)


def test_simplex_norm_0():
    x = np.array([1.1, 0.6])
    x_proj, d_0, theta = simplex_proj(x)
    assert d_0 == l0_norm(x_proj)


def test_simplex_exterior_to_frontier():
    # exterior point projects to frontier
    x = np.array([0.6, 1.1])
    x_proj, d_0, theta = simplex_proj(x)
    assert_almost_equal(np.sum(x_proj), 1.0, decimal=8)
    assert d_0 == l0_norm(x_proj)


def test_simplex_correct_endpoint_y():
    # assert
    x = np.array([0.0, 1.1])
    x_proj, d_0, theta = simplex_proj(x)
    assert_almost_equal(np.array([0.0, 1.0]), x_proj, decimal=8)
    assert d_0 == l0_norm(x_proj)


def test_simplex_correct_endpoint_x():
    x = np.array([1.1, 0.0])
    x_proj, d_0, theta = simplex_proj(x)
    assert_almost_equal(np.array([1.0, 0.0]), x_proj, decimal=8)
    assert d_0 == l0_norm(x_proj)


def test_simplex_inside():
    # assert inside simplex identical
    for i in range(100):
        x = np.random.rand(100)
        x_proj, d_0, theta = simplex_proj(x)
        assert_almost_equal(x, x_proj, decimal=8)
        assert d_0 == l0_norm(x_proj)


def test_ball_inside():
    # assert inside B(z) identical
    for i in range(100):
        radius = 10
        x = np.random.rand(100) * radius
        x_proj, d_0, theta = l1_ball_proj(x, radius)
        assert_almost_equal(x, x_proj, decimal=8)
        assert 100 == l0_norm(x_proj)


def test_ball_correct():
    # assert
    x = np.array([0.0, 1.1])
    x_proj, d_0, theta = l1_ball_proj(x, 1)
    assert_almost_equal(np.array([0.0, 1.0]), x_proj, decimal=8)
    assert d_0 == l0_norm(x_proj)


def test_ball_correct_x():
    x = np.array([1.1, 0.0])
    x_proj, d_0, theta = l1_ball_proj(x, 1)
    assert_almost_equal(np.array([1.0, 0.0]), x_proj, decimal=8)
    assert d_0 == l0_norm(x_proj)


def test_ball_correct_x_y():
    x = np.array([1.1, 0.1])
    x_proj, d_0, theta = l1_ball_proj(x, 1)
    assert_almost_equal(np.array([1.0, 0.0]), x_proj, decimal=8)
    assert d_0 == l0_norm(x_proj)
