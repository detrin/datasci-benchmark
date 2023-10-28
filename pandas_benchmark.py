import time
import pandas as pd
import numpy as np
import argparse
import os
import tempfile


def benchmark_pandas(n_rows=100000, n_cols=20, verbose=True):
    # Create a DataFrame with random data
    if verbose:
        print(f"Creating a DataFrame with {n_rows} rows and {n_cols} columns...")
    df = pd.DataFrame(
        np.random.randint(0, 100, size=(n_rows, n_cols)),
        columns=[f"col{i}" for i in range(n_cols)],
    )

    # Perform some typical DataFrame operations and measure the time taken
    if verbose:
        print("Performing operations...")

    start_time = time.time()

    df_sorted = df.sort_values("col0")
    df_grouped = df.groupby("col0").mean()
    df["new_col"] = df["col0"].apply(lambda x: x * 2)

    end_time = time.time()

    # Write the DataFrame to a CSV file and measure the time taken
    if verbose:
        print("Writing DataFrame to CSV...")
    start_time_csv = time.time()
    with tempfile.TemporaryDirectory() as tmpdirname:
        df.to_csv(os.path.join(tmpdirname, "data.csv"), index=False)
    end_time_csv = time.time()

    # Return the time taken for operations and writing to CSV
    operations_time = end_time - start_time
    csv_time = end_time_csv - start_time_csv
    if verbose:
        print(f"Time for operations: {operations_time} seconds")
        print(f"Time for writing to CSV: {csv_time} seconds")

    results = {"operations_time": operations_time, "csv_time": csv_time}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows", type=int, default=100000, help="Number of rows in the DataFrame"
    )
    parser.add_argument(
        "--cols", type=int, default=20, help="Number of columns in the DataFrame"
    )
    args = parser.parse_args()

    benchmark_pandas(n_rows=args.rows, n_cols=args.cols)
