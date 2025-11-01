import pandas as pd
import joblib

def detect_anomalies(df):
    """
    Predict anomalies using the trained model.
    """
    model = joblib.load("../models/trained_model.pkl")
    features = ['HeartRate', 'BloodPressure', 'OxygenSaturation', 
                'Heart_to_BP_Ratio', 'OxygenVariation']
    df['Anomaly'] = model.predict(df[features])
    df['Anomaly'] = df['Anomaly'].map({1: "Normal", -1: "Abnormal"})
    print(f"✅ Detected {sum(df['Anomaly'] == 'Abnormal')} abnormalities. - detect_anomalies.py:13")
    return df

if __name__ == "__main__":
    from feature_engineering import feature_engineering
    from preprocess_data import preprocess_data
    from load_data import load_data

    df = feature_engineering(preprocess_data(load_data("../data/medical_data.csv")))
    results = detect_anomalies(df)
    results.to_csv("../data/results.csv", index=False)
