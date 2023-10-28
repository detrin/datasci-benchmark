import time
from optbinning import OptimalBinning   
import argparse
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import warnings  
warnings.filterwarnings("ignore")  

  
def benchmark_binning(n_rows=100000, n_cols=20, n_classes=2, verbose=True):  
    # Generate a random n-class classification problem  
    if verbose:  
        print(f"Generating dataset with {n_rows} samples, {n_cols} features...")  
    X, y = make_classification(  
        n_samples=n_rows * 2, n_features=n_cols, n_classes=n_classes  
    )  
    X_train, X_test, y_train, y_test = train_test_split(  
        X, y, random_state=42, test_size=0.5  
    )  
  
    total_train_time = 0  
    total_transform_time = 0  
  
    for i in range(n_cols):  
        # Define OptimalBinning object  
        optb = OptimalBinning(name=f"feature_{i}", dtype="numerical", solver="cp")  
  
        # Train the model and measure the time it takes  
        start_time = time.time()  
        optb.fit(X_train[:, i], y_train)  
        end_time = time.time()  
  
        # Measure the time it takes to transform the data  
        start_time_pred = time.time()  
        transformed_data = optb.transform(X_test[:, i])  
        end_time_pred = time.time()  
  
        # Accumulate the training and transformation times  
        total_train_time += end_time - start_time  
        total_transform_time += end_time_pred - start_time_pred  
  
    if verbose:  
        print(f"Total training time: {total_train_time} seconds")  
        print(f"Total transformation time: {total_transform_time} seconds")  
  
    return {"total_train_time": total_train_time, "total_transform_time": total_transform_time}  
  
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()  
    parser.add_argument(  
        "--rows", type=int, default=100000, help="Number of rows in the dataset"  
    )  
    parser.add_argument(  
        "--cols", type=int, default=20, help="Number of columns in the dataset"  
    )  
    args = parser.parse_args()  
  
    benchmark_optbinning(n_rows=args.rows, n_cols=args.cols)  
  
