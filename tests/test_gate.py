"""Projection gate tests: sums to 1, caps+floor respected, identity if none flagged."""
import numpy as np
from decision.gate import project
def test_gate_identity():
    a = np.array([0.7, 0.3])
    np.testing.assert_allclose(project(a, {0: 0.8}, {0: False}), a, atol=1e-6)
def test_gate_cap():
    q = project(np.array([0.8, 0.2]), {0: 0.5}, {0: True})
    assert abs(q.sum()-1) < 1e-6 and q[0] <= 0.5 + 1e-6
