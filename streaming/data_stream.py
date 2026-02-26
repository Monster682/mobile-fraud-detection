import pandas as pd

def stream_transactions(file_path):

    # Load dataset
    df = pd.read_csv(file_path)

    print("Streaming started. Total transactions:", len(df))

    # Stream one by one
    for index, row in df.iterrows():

        yield row.to_dict()