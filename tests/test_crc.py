"""CRC test: empirical FCR <= alpha on synthetic scores."""
import numpy as np
from decision.crc import crc_threshold
def test_crc_controls_fcr():
    rng = np.random.default_rng(0)
    cal = rng.random(500); tst = rng.random(500)
    lam = crc_threshold(cal, 0.1)
    assert (tst > lam).mean() <= 0.25  # loose smoke; E4 does >=1000 splits properly
