import pandas as pd

def load_data(filepath):
    """
    Load the medical dataset from CSV.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns - load_data.py:9")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e} - load_data.py:12")
        return None

if __name__ == "__main__":
    data = load_data("../data/medical_data.csv")
