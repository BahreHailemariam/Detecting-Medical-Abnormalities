import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Preprocess medical data:
    - Handle missing values
    - Encode categorical variables
    - Scale numerical features
    """
    df = df.dropna()
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print("✅ Data preprocessing completed. - preprocess_data.py:15")
    return df

if __name__ == "__main__":
    from load_data import load_data
    df = load_data("../data/medical_data.csv")
    clean_df = preprocess_data(df)
