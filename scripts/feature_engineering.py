import pandas as pd
import numpy as np

def feature_engineering(df):
    """
    Create derived features for anomaly detection.
    Example: Vital rate ratios, averages, and interaction terms.
    """
    df['Heart_to_BP_Ratio'] = df['HeartRate'] / (df['BloodPressure'] + 1)
    df['OxygenVariation'] = np.abs(df['OxygenSaturation'] - df['OxygenSaturation'].mean())
    print("✅ Feature engineering complete. - feature_engineering.py:11")
    return df

if __name__ == "__main__":
    from preprocess_data import preprocess_data
    from load_data import load_data
    df = preprocess_data(load_data("../data/medical_data.csv"))
    feature_df = feature_engineering(df)
