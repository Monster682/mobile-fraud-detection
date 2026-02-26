import time
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

from streaming.data_stream import stream_transactions
from src.feature_engineering import create_features
from src.preprocessing import preprocess


BATCH_SIZE = 500   # increase batch size for faster processing


def classify_latency(latency):

    if latency < 0.005:
        return "LOW"

    elif latency < 0.02:
        return "MEDIUM"

    else:
        return "HIGH"


def run_streaming_pipeline():

    print("Loading models...")

    autoencoder = load_model("models/autoencoder.keras", compile=False)

    iso = joblib.load("models/isolation_forest.pkl")

    print("Models loaded successfully")

    results = []

    batch = []

    total_count = 0

    start_total = time.time()

    for transaction in stream_transactions("data/transactions.csv"):

        batch.append(transaction)

        if len(batch) == BATCH_SIZE:

            process_batch(batch, autoencoder, iso, results)

            total_count += len(batch)

            print("Processed:", total_count)

            batch = []

    # process remaining
    if batch:

        process_batch(batch, autoencoder, iso, results)

        total_count += len(batch)

    final = pd.concat(results)

    final.to_csv("outputs/streaming_predictions.csv", index=False)

    total_time = time.time() - start_total

    print("\nStreaming completed successfully")

    print("Total transactions processed:", total_count)

    print("Total time:", total_time, "seconds")

    print("Average latency:", final["latency"].mean())


def process_batch(batch, autoencoder, iso, results):

    start = time.time()

    df = pd.DataFrame(batch)

    df = create_features(df)

    X, df_clean, scaler = preprocess(df)

    recon = autoencoder.predict(X, verbose=0)

    mse = np.mean((X - recon)**2, axis=1)

    ae_pred = (mse > 0.01).astype(int)

    iso_pred = iso.predict(X)

    iso_pred = np.where(iso_pred == -1, 1, 0)

    fraud = np.where((ae_pred + iso_pred) > 0, 1, 0)

    latency = (time.time() - start) / len(batch)

    latency_level = classify_latency(latency)

    df_clean["fraud"] = fraud

    df_clean["latency"] = latency

    df_clean["latency_level"] = latency_level

    results.append(df_clean)


if __name__ == "__main__":

    run_streaming_pipeline()