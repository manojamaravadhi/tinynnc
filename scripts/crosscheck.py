#!/usr/bin/env python3
"""Cross-check TinyNNC against NumPy.

For each configuration: generate a random model and input, compute the expected
output in NumPy, run ./tinynnc with each --opt setting, and compare.

Pass criteria:
  * C vs NumPy: np.allclose(rtol=1e-5, atol=1e-5). NumPy's matmul (BLAS) adds
    numbers in a different order than TinyNNC, so the results are not bit-identical.
  * C optimized vs C unoptimized: max |diff| <= 1e-5 (expected: exactly 0).
Exits non-zero on any failure.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_mlp import generate  # noqa: E402

# (label, dims, batch, activations, softmax, redundant)
CONFIGS = [
    ("tiny relu",          [4, 3, 2],               1, ["relu", "none"], False, False),
    ("deep relu+softmax",  [16, 32, 32, 32, 10],    2, ["relu"] * 3 + ["none"], True, False),
    ("sigmoid mlp",        [8, 16, 4],              3, ["sigmoid", "none"], False, False),
    ("mixed acts",         [12, 24, 24, 6],         4, ["relu", "sigmoid", "none"], True, False),
    ("redundant ops",      [8, 16, 16, 4],          2, ["relu", "relu", "none"], True, True),
    ("redundant sigmoid",  [10, 20, 5],             3, ["sigmoid", "none"], False, True),
    ("wide 64x128x64",     [64, 128, 64],           8, ["relu", "none"], False, False),
    ("wide 256x512x256",   [256, 512, 256],         4, ["relu", "none"], True, True),
]


def run_c(binary, model, inp, opt):
    out = subprocess.run([binary, model, "--input", inp, "--opt", opt, "--raw"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("tinynnc failed (%s): %s" % (opt, out.stderr.strip()))
    return np.array(out.stdout.split(), dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="./tinynnc")
    ap.add_argument("--opts", default="none,all",
                    help="comma list of --opt settings to test; the first one is the baseline")
    a = ap.parse_args()
    opts = a.opts.split(",")

    failures = 0
    print("== crosscheck vs NumPy %s ==" % np.__version__)
    print("  %-20s %-10s %-14s %-14s %s" % ("model", "opt", "max|C-numpy|", "max|opt-none|",
                                             "result"))
    with tempfile.TemporaryDirectory() as tmp:
        for seed, (label, dims, batch, acts, sm, red) in enumerate(CONFIGS):
            text, x, expected = generate(dims, batch, acts, sm, red, seed + 100)
            model = os.path.join(tmp, "m%d.txt" % seed)
            inp = os.path.join(tmp, "x%d.txt" % seed)
            with open(model, "w") as f:
                f.write(text)
            np.savetxt(inp, x.reshape(1, -1), fmt="%.9g")
            exp = expected.reshape(-1).astype(np.float64)
            base = None
            for opt in opts:
                try:
                    got = run_c(a.bin, model, inp, opt)
                except RuntimeError as e:
                    print("  %-20s %-10s ERROR: %s" % (label, opt, e))
                    failures += 1
                    continue
                if got.shape != exp.shape:
                    print("  %-20s %-10s shape mismatch %s vs %s" % (label, opt, got.shape,
                                                                     exp.shape))
                    failures += 1
                    continue
                d_np = np.max(np.abs(got - exp))
                ok = np.allclose(got, exp, rtol=1e-5, atol=1e-5)
                if base is None:
                    base = got
                    d_opt = "-"
                else:
                    dd = np.max(np.abs(got - base))
                    d_opt = "%.3g" % dd
                    ok = ok and dd <= 1e-5
                failures += 0 if ok else 1
                print("  %-20s %-10s %-14.3g %-14s %s" % (label, opt, d_np, d_opt,
                                                          "PASS" if ok else "FAIL"))
    total = len(CONFIGS) * len(opts)
    print("crosscheck: %d/%d passed -> %s" % (total - failures, total,
                                               "OK" if failures == 0 else "FAIL"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
