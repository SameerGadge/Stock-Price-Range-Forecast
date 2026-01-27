import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import numpy as np

# --- CUSTOM LOSS FUNCTION ---
# This forces the Neural Network to predict the "edge" (quantile) 
# instead of the average.
def quantile_loss(q):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
    return loss

class DeepQuantileModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None

    def build_model(self, quantile):
        model = Sequential([
            Input(shape=self.input_shape),
            # LSTM Layer: Learn patterns over time
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss=quantile_loss(quantile))
        return model

    def train(self, X_train, y_train, epochs=20, batch_size=32):
        # We need to reshape data for LSTM: [Samples, TimeSteps, Features]
        X_reshaped = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
        
        # Train Lower Bound (Bear)
        self.model_low = self.build_model(0.05)
        self.model_low.fit(X_reshaped, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Train Upper Bound (Bull)
        self.model_high = self.build_model(0.95)
        self.model_high.fit(X_reshaped, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        return self.model_low, self.model_high

    def predict(self, X):
        # Reshape input to match LSTM expectation
        X_reshaped = X.values.reshape((X.shape[0], 1, X.shape[1]))
        
        pred_low = self.model_low.predict(X_reshaped, verbose=0).flatten()
        pred_high = self.model_high.predict(X_reshaped, verbose=0).flatten()
        
        return pred_low, pred_high