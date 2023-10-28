import time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class SimpleNet(nn.Module):
    def __init__(self, n_features=20):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(n_features, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def benchmark_torch(n_rows=100000, n_cols=20, n_classes=2, verbose=True):
    # Create a random n-class classification problem
    if verbose:
        print(f"Generating dataset with {n_rows} samples, {n_cols} features...")
    X, y = make_classification(
        n_samples=n_rows * 2, n_features=n_cols, n_classes=n_classes
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.5
    )
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()

    model = SimpleNet(n_features=n_cols)

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train the model and measure the time it takes
    if verbose:
        print("Training model...")
    start_time = time.time()
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i in range(n_rows):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(X_train[i])
            loss = criterion(outputs.view(1, -1), torch.tensor([y_train[i]]))
            loss.backward()
            optimizer.step()

    end_time = time.time()

    # Measure the time it takes to make predictions
    if verbose:
        print("Making predictions...")
    start_time_pred = time.time()
    with torch.no_grad():
        predictions = model(X_test)
    end_time_pred = time.time()

    # Return the time it took to train the model and to make predictions
    train_time = end_time - start_time
    pred_time = end_time_pred - start_time_pred
    if verbose:
        print(f"Time for training: {train_time} seconds")
        print(f"Time for making predictions: {pred_time} seconds")
    results = {"train_time": train_time, "pred_time": pred_time}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows", type=int, default=100000, help="Number of rows in the dataset"
    )
    parser.add_argument(
        "--cols", type=int, default=20, help="Number of columns in the dataset"
    )
    args = parser.parse_args()

    benchmark_torch(n_rows=args.rows, n_cols=args.cols)
