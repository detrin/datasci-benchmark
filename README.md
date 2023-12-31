# Data Science Benchmarks
Benchmark your server for data science tasks.

## Usage
```bash
git clone https://github.com/detrin/datasci-benchmark
cd datasci-benchmark
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```
Then run particular benchmark file with
```bash
python pandas_benchmark.py --rows 1000 --cols 10
```
or run all benchmarks with
```bash
python benchmark.py --rows 1000 --cols 10 --runs 10
```

## My options
```bash
python benchmark.py --rows 10000000 --cols 10 --runs 20 --json --pandas --polars --xgboost --lightgbm --name M1
python benchmark.py --rows 1000000 --cols 10 --runs 50 --json --binning --logreg --name M1
python benchmark.py --rows 100000 --cols 10 --runs 50 --json --knn --name M1
python benchmark.py --rows 10000 --cols 10 --runs 50 --json --svm --torch --name M1
```
