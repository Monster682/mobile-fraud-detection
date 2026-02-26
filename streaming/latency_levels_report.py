import pandas as pd


def classify_latency(latency):

    if latency < 0.005:
        return "LOW"

    elif latency < 0.02:
        return "MEDIUM"

    else:
        return "HIGH"


def generate_latency_report():

    print("\nGenerating Latency Report...\n")

    # Load streaming output
    df = pd.read_csv("outputs/streaming_predictions.csv")

    # If latency_level column not exists, create it
    if "latency_level" not in df.columns:

        df["latency_level"] = df["latency"].apply(classify_latency)

    total = len(df)

    avg_latency = df["latency"].mean()

    max_latency = df["latency"].max()

    min_latency = df["latency"].min()

    distribution = df["latency_level"].value_counts()

    print("===== LATENCY REPORT =====\n")

    print("Total transactions:", total)

    print("\nAverage latency:", avg_latency, "seconds")

    print("Max latency:", max_latency, "seconds")

    print("Min latency:", min_latency, "seconds")

    print("\nLatency Level Distribution:")

    print(distribution)

    # Save report file
    df.to_csv("outputs/streaming_predictions_with_latency_levels.csv", index=False)

    print("\nReport saved to outputs/streaming_predictions_with_latency_levels.csv")


if __name__ == "__main__":

    generate_latency_report()