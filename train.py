"""Train a simple linear regression model for disease prediction."""

import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def train(train_data_path, model_path):
    """Train the model on the provided data.

    Parameters
    ----------
    train_data_path : str
        Path to the training data CSV file.
    model_path : str
        Path where the trained model will be saved.
    """
    df = pd.read_csv(train_data_path)
    features = df[["rainfall"]].fillna(0)
    target = df["disease_cases"].fillna(0)

    model = RandomForestRegressor()
    model.fit(features, target)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a disease prediction model")
    parser.add_argument("train_data", help="Path to training data CSV file")
    parser.add_argument("model", help="Path to save the trained model")
    args = parser.parse_args()

    train(args.train_data, args.model)
