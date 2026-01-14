"""Generate predictions using a trained model."""

import argparse

import joblib
import pandas as pd


def predict(model_path, historic_data_path, future_data_path, out_file_path):
    """Generate predictions using the trained model.

    Parameters
    ----------
    model_path : str
        Path to the trained model file.
    historic_data_path : str
        Path to historic data CSV file (unused in this simple model).
    future_data_path : str
        Path to future climate data CSV file.
    out_file_path : str
        Path where predictions will be saved.
    """
    model = joblib.load(model_path)
    future_df = pd.read_csv(future_data_path)
    features = future_df[["rainfall"]].fillna(0)

    predictions = model.predict(features)
    output_df = future_df[["time_period", "location"]].copy()
    output_df["sample_0"] = predictions
    output_df.to_csv(out_file_path, index=False)
    print(f"Predictions saved to {out_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate disease predictions")
    parser.add_argument("model", help="Path to trained model file")
    parser.add_argument("historic_data", help="Path to historic data CSV file")
    parser.add_argument("future_data", help="Path to future climate data CSV file")
    parser.add_argument("out_file", help="Path to save predictions CSV file")
    args = parser.parse_args()

    predict(args.model, args.historic_data, args.future_data, args.out_file)
