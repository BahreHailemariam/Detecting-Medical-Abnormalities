from sklearn.ensemble import IsolationForest
import joblib

def train_anomaly_model(df):
    """
    Train anomaly detection model (Isolation Forest).
    """
    features = ['HeartRate', 'BloodPressure', 'OxygenSaturation', 
                'Heart_to_BP_Ratio', 'OxygenVariation']
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(df[features])
    joblib.dump(model, "../models/trained_model.pkl")
    print("✅ Model trained and saved successfully. - train_model.py:13")

if __name__ == "__main__":
    from feature_engineering import feature_engineering
    from preprocess_data import preprocess_data
    from load_data import load_data

    df = feature_engineering(preprocess_data(load_data("../data/medical_data.csv")))
    train_anomaly_model(df)
