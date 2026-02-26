import pandas as pd
import joblib

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from src.feature_engineering import create_features
from src.preprocessing import preprocess


df = pd.read_csv("data/transactions.csv")

df = create_features(df)

X, df_clean, scaler = preprocess(df)

input_dim = X.shape[1]

input_layer = Input(shape=(input_dim,))

encoded = Dense(16, activation="relu")(input_layer)

decoded = Dense(input_dim, activation="sigmoid")(encoded)

autoencoder = Model(input_layer, decoded)

autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.fit(X, X, epochs=5, batch_size=32)

autoencoder.save("models/autoencoder.keras")


from sklearn.ensemble import IsolationForest

iso = IsolationForest()

iso.fit(X)

joblib.dump(iso, "models/isolation_forest.pkl")

print("Models trained successfully")