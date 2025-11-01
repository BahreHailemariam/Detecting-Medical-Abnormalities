import matplotlib.pyplot as plt
import pandas as pd

def plot_results(df):
    """
    Visualize the detected anomalies in the dataset.
    """
    normal = df[df['Anomaly'] == "Normal"]
    abnormal = df[df['Anomaly'] == "Abnormal"]

    plt.figure(figsize=(10,6))
    plt.scatter(normal.index, normal['HeartRate'], color='blue', label="Normal")
    plt.scatter(abnormal.index, abnormal['HeartRate'], color='red', label="Abnormal")
    plt.title("Detected Medical Abnormalities (Heart Rate)")
    plt.xlabel("Index")
    plt.ylabel("Heart Rate")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("../data/results.csv")
    plot_results(df)
