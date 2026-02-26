import pandas as pd
import numpy as np

def create_features(df):

    if "amount" in df.columns:
        df["amount_log"] = df["amount"].apply(
            lambda x: 0 if x <= 0 else np.log(x)
        )

    df = df.fillna(0)

    return df