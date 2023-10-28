# Data Science Benchmarks
Benchmark your server for data science tasks.

## Usage
```bash
git clone https://github.com/detrin/datasci-benchmark
cd datasci-benchmark
python -m venv env
source env/bin/activate
pip install -r requirements.txt

python pandas_benchmark.py --rows 1000 --cols 10
python polars_benchmark.py --rows 1000 --cols 10
python xgboost_benchmark.py --rows 1000 --cols 10
python torch_benchmark.py --rows 1000 --cols 10

python benchmark.py --rows 1000 --cols 10 --runs 10
python benchmark.py --rows 1000000 --cols 10 --runs 50 --pandas --polars --xgboost
```