from sklearn.preprocessing import StandardScaler

def preprocess(df):

    df = df.select_dtypes(include=["number"])

    scaler = StandardScaler()

    X = scaler.fit_transform(df)

    return X, df, scaler