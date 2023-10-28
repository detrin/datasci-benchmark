import numpy as np
import timeit
import argparse
import warnings  
import json

warnings.filterwarnings("ignore")  

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
parser.add_argument("--binning", action="store_true", help="Benchmark binning")
parser.add_argument("--logreg", action="store_true", help="Benchmark Logistic Regression")
parser.add_argument("--xgboost", action="store_true", help="Benchmark xgboost")
parser.add_argument("--lightgbm", action="store_true", help="Benchmark lightgbm")
parser.add_argument("--torch", action="store_true", help="Benchmark torch")
parser.add_argument("--svm", action="store_true", help="Benchmark SVM")
parser.add_argument("--knn", action="store_true", help="Benchmark KNN")

parser.add_argument("--json", action="store_true", help="Output results as JSON")
parser.add_argument("--name", type=str, default="mymachine", help="Name of the machine")
args = parser.parse_args()

if args.runs < 3:
    raise ValueError("Number of runs must be at least 3")

test_info = []
if args.pandas:
    from pandas_benchmark import benchmark_pandas
    test_info.append(["pandas", benchmark_pandas])
if args.polars:
    from polars_benchmark import benchmark_polars
    test_info.append(["polars", benchmark_polars])
if args.binning:
    from binning_benchmark import benchmark_binning
    test_info.append(["binning", benchmark_binning])
if args.logreg:
    from logreg_benchmark import benchmark_logreg
    test_info.append(["logreg", benchmark_logreg])
if args.xgboost:
    from xgboost_benchmark import benchmark_xgboost
    test_info.append(["xgboost", benchmark_xgboost])
if args.lightgbm:
    from lightgbm_benchmark import benchmark_lightgbm
    test_info.append(["lightgbm", benchmark_lightgbm])
if args.torch:
    from torch_benchmark import benchmark_torch
    test_info.append(["torch", benchmark_torch])
if args.svm:
    from svm_benchmark import benchmark_svm
    test_info.append(["svm", benchmark_svm])
if args.knn:
    from knn_benchmark import benchmark_knn
    test_info.append(["knn", benchmark_knn])


for name, func in test_info:
    if not args.json:
        print(f"Benchmarking {name}...")
    times = []
    for _ in range(args.runs):
        times.append(func(n_rows=args.rows, n_cols=args.cols, verbose=False))

    keys = times[0].keys()
    mus = {k: np.mean([t[k] for t in times]) for k in keys}
    stds = {k: np.std([t[k] for t in times]) for k in keys}

    if not args.json:
        for k in keys:
            print(f"Mean {k}: {mus[k]:.4f} s +/- {stds[k]:.4f} s")
    else:
        results = []
        for k in keys:
            results.append({"machine": args.name, "name": name, "metric": k, "value": mus[k], "std": stds[k]})

if args.json:
    print(json.dumps(results))