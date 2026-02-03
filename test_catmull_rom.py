import numpy as np
import pytest

from catmull_rom import catrom


def f(t):
    return (t**3 + 5 + 2*t**1)


x = np.arange(0, 100)
y = f(x)


def test_scalar_far_from_edge():
    t = np.array([5.2, 7.8, 9.7])
    yinterp = catrom(t, y)
    yexact = f(t)
    assert yinterp == pytest.approx(yexact, rel=0.001)


def test_scalar_near_left_edge():
    t = np.array([0.2, 0.1, 0.9])
    yinterp = catrom(t, y)
    yexact = f(t)
    assert yinterp == pytest.approx(yexact, rel=0.1)

def test_scalar_near_right_edge():
    t = np.array([99.3, 99.7])
    yinterp = catrom(t, y)
    yexact = f(t)
    assert yinterp == pytest.approx(yexact, rel=0.001)
