import time  
import xgboost as xgb  
import argparse  
from sklearn.datasets import make_classification  
from sklearn.model_selection import train_test_split  
  
def benchmark_xgboost(n_rows=100000, n_cols=20, n_classes=2, verbose=True):  
    # Generate a random n-class classification problem  
    if verbose:
        print(f"Generating dataset with {n_rows} samples, {n_cols} features...")  
    X, y = make_classification(n_samples=n_rows*2, n_features=n_cols, n_classes=n_classes)  
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)  
  
    # Create a DMatrix (data structure used by XGBoost)  
    dtrain = xgb.DMatrix(X_train, label=y_train)  
    dtest = xgb.DMatrix(X_test, label=y_test)  
  
    # Define parameters for the XGBoost classifier  
    param = {  
        'max_depth': 3,  
        'eta': 0.3,  
        'objective': 'multi:softprob',  
        'num_class': n_classes  
    }  
    num_round = 20  # the number of training iterations  
  
    # Train the model and measure the time it takes  
    if verbose:
        print("Training model...")  
    start_time = time.time()  
    bst = xgb.train(param, dtrain, num_round)  
    end_time = time.time()  
  
    # Measure the time it takes to make predictions  
    if verbose:
        print("Making predictions...")  
    start_time_pred = time.time()  
    predictions = bst.predict(dtest)  
    end_time_pred = time.time()  
  
    # Return the time it took to train the model and to make predictions  
    train_time = end_time - start_time
    pred_time = end_time_pred - start_time_pred
    if verbose:
        print(f"Training time: {train_time} seconds")  
        print(f"Prediction time: {pred_time} seconds")
    results = {
        'train_time': train_time,
        'pred_time': pred_time
    }
    return results
  
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()  
    parser.add_argument('--rows', type=int, default=100000,  
                        help='Number of rows in the dataset')  
    parser.add_argument('--cols', type=int, default=20,  
                        help='Number of columns in the dataset')  
    args = parser.parse_args()  
  
    benchmark_xgboost(n_rows=args.rows, n_cols=args.cols)  
