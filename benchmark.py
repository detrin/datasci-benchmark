import numpy as np
import timeit

from pandas_benchmark import benchmark_pandas
from polars_benchmark import benchmark_polars
from torch_benchmark import benchmark_pytorch
from xgboost_benchmark import benchmark_xgboost

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--rows", type=int, default=1000, help="Number of rows in the DataFrame"
)
parser.add_argument(
    "--cols", type=int, default=10, help="Number of columns in the DataFrame"
)
parser.add_argument(
    "--runs", type=int, default=100, help="Number of experiments to run"
)

parser.add_argument("--pandas", action="store_true", help="Benchmark pandas")
parser.add_argument("--polars", action="store_true", help="Benchmark polars")
parser.add_argument("--xgboost", action="store_true", help="Benchmark xgboost")
parser.add_argument("--pytorch", action="store_true", help="Benchmark pytorch")
args = parser.parse_args()

test_info = []
if args.pandas:
    test_info.append(["pandas", benchmark_pandas])
if args.polars:
    test_info.append(["polars", benchmark_polars])
if args.xgboost:
    test_info.append(["xgboost", benchmark_xgboost])
if args.pytorch:
    test_info.append(["pytorch", benchmark_pytorch])


for name, func in test_info:
    print(f"Benchmarking {name}...")
    times = []
    for _ in range(args.runs):
        times.append(func(n_rows=args.rows, n_cols=args.cols, verbose=False))

    keys = times[0].keys()
    mus = {k: np.mean([t[k] for t in times]) for k in keys}
    stds = {k: np.std([t[k] for t in times]) for k in keys}

    for k in keys:
        print(f"Mean {k}: {mus[k]:.4f} s +/- {stds[k]:.4f} s")
