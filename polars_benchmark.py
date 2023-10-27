import time  
import polars as pl  
import numpy as np  
import argparse  
import os
  
def benchmark_polars(n_rows=100000, n_cols=20):  
    # Create a DataFrame with random data  
    print(f"Creating a DataFrame with {n_rows} rows and {n_cols} columns...")  
    df = pl.DataFrame({f'col{i}': np.random.randint(0, 100, size=n_rows) for i in range(n_cols)})  
  
    # Perform some typical DataFrame operations and measure the time taken  
    print("Performing operations...")  
    start_time = time.time()  
  
    df_sorted = df.sort('col0')  
    df_grouped = df.group_by('col0').agg(pl.mean('col1'))  
    df = df.with_columns(df['col0'] * 2)  
  
    end_time = time.time()  
  
    # Write the DataFrame to a CSV file and measure the time taken  
    print("Writing DataFrame to CSV...")  
    start_time_csv = time.time()  
    df.write_csv('data.csv')  
    end_time_csv = time.time() 
    os.remove('data.csv') 
  
    # Return the time taken for operations and writing to CSV  
    return end_time - start_time, end_time_csv - start_time_csv  
  
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()  
    parser.add_argument('--rows', type=int, default=100000,  
                        help='Number of rows in the DataFrame')  
    parser.add_argument('--cols', type=int, default=20,  
                        help='Number of columns in the DataFrame')  
    args = parser.parse_args()  
  
    operations_time, csv_time = benchmark_polars(n_rows=args.rows, n_cols=args.cols)  
    print(f"Time for operations: {operations_time} seconds")  
    print(f"Time for writing to CSV: {csv_time} seconds")  
