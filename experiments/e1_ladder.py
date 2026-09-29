"""E1 diagnostic ladder (Gate 1): rho -> budget -> invariance -> trust -> anchor -> U -> calibration.

Reads config, runs R1-R5 x rho_corr x 5 paired seeds, writes results table + cert logs.
Usage: python -m experiments.e1_ladder --config configs/d1_budget.yaml --seeds 0 1 2 3 4
"""
import argparse
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", nargs="*", default=[0]); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); print(f"E1 smoke={a.smoke} config={a.config} (Step 6 stub)")
if __name__ == "__main__": main()
