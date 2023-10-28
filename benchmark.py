
import numpy as np  
import timeit  

from pandas_benchmark import benchmark_pandas
from polars_benchmark import benchmark_polars
from torch_benchmark import benchmark_pytorch
from xgboost_benchmark import benchmark_xgboost

import argparse

parser = argparse.ArgumentParser()  
parser.add_argument('--rows', type=int, default=1000,  
                    help='Number of rows in the DataFrame')  
parser.add_argument('--cols', type=int, default=10,  
                    help='Number of columns in the DataFrame')  
parser.add_argument('--runs', type=int, default=100,  
                    help='Number of experiments to run')  
args = parser.parse_args()  
  
test_info = [
    ["pandas", benchmark_pandas],
    ["polars", benchmark_polars],
    ["xgboost", benchmark_xgboost],
    ["pytorch", benchmark_pytorch]
]  

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

