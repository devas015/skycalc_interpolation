import numpy as np
import pytest
import math
from catmull_rom import catrom
from hypothesis import given, strategies as st
import pytest
import math

#Using Hypothesis to Test
def sinc(t):
    return np.cos(t) * 4 + 1

@given(x=st.floats(allow_nan=False, allow_infinity=False))
def test_sinc(x):
    assert (sinc(x) - 1)/4 == pytest.approx(math.cos(x))

#Individual Test Cases
def f(t):
    return (np.sin(t))
x = np.arange(0, 100)
y = f(x)

def test_scalar_far_from_edge():
    t = np.array([5.2, 7.8, 9.7])
    yinterp = catrom(t, y)
    yexact = f(t)
    assert yinterp == pytest.approx(yexact, rel=0.05)


def test_scalar_near_left_edge():
    t = np.array([0.2, 0.1, 0.9])
    yinterp = catrom(t, y)
    yexact = f(t)
    assert yinterp == pytest.approx(yexact, rel=0.15)

def test_scalar_near_right_edge():
    t = np.array([98.5, 98.6])
    yinterp = catrom(t, y)
    yexact = f(t)
    assert yinterp == pytest.approx(yexact, rel=0.15)
