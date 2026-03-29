import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Tuple

SEQ_LEN = 5  # Use past 5 days for sequence modeling

class AdvancedDemandModel(nn.Module):
    """
    An advanced LSTM-based demand model, mimicking aspects of Temporal Fusion Transformers (TFTs)
    by explicitly handling various input types (e.g., current price, competitor price, seasonality, promo).
    A full TFT would involve more complex components like static covariates, attention mechanisms,
    and quantile outputs, but this provides a DFL-aware foundation.
    """
    def __init__(self):
        super().__init__()
        # Input features: price, competitor_price, seasonality, promo
        self.lstm = nn.LSTM(input_size=4, hidden_size=64, batch_first=True) # Increased hidden size
        self.fc = nn.Linear(64, 1) # Output a single demand value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the demand model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, SEQ_LEN, input_size).
                              input_size = 4 (price, competitor_price, seasonality, promo)

        Returns:
            torch.Tensor: Predicted demand tensor of shape (batch_size, 1).
        """
        out, _ = self.lstm(x)
        # Use the output from the last time step
        out = out[:, -1, :]
        return self.fc(out)

def create_sequences(df_sku: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates time-series sequences from SKU-specific DataFrame for LSTM training.

    Args:
        df_sku (pd.DataFrame): DataFrame for a single SKU, containing demand and features.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X (features sequences), y (target demands).
    """
    X, y = [], []
    df_sku = df_sku.sort_values("day")
    # Features: price, competitor_price, seasonality, promo
    features = df_sku[["price", "competitor_price", "seasonality", "promo"]].values
    target = df_sku["demand"].values

    for i in range(len(df_sku) - SEQ_LEN):
        X.append(features[i : i + SEQ_LEN])
        y.append(target[i + SEQ_LEN]) # Predict demand for the day after the sequence ends
    return np.array(X), np.array(y)

def train_models(df: pd.DataFrame) -> Dict[str, AdvancedDemandModel]:
    """
    Trains an AdvancedDemandModel for each unique SKU in the DataFrame.

    Args:
        df (pd.DataFrame): The full dataset containing demand and features for all SKUs.

    Returns:
        Dict[str, AdvancedDemandModel]: A dictionary mapping SKU to its trained demand model.
    """
    models = {}
    for sku in df["sku"].unique():
        df_sku = df[df["sku"] == sku]
        X, y = create_sequences(df_sku)

        if len(X) == 0:
            print(f"Skipping SKU {sku}: Not enough data to create sequences.")
            continue

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        model = AdvancedDemandModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        loss_fn = nn.MSELoss() # Standard MSE loss for demand prediction

        # Training loop
        for epoch in range(100):
            model.train()
            pred = model(X_tensor)
            loss = loss_fn(pred, y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if (epoch + 1) % 20 == 0:
            #     print(f"SKU {sku}, Epoch {epoch+1}, Loss: {loss.item():.4f}")

        model.eval() # Set model to evaluation mode after training
        models[sku] = model
    return models

def predict_demand(
    model: AdvancedDemandModel,
    price: float,
    competitor_price: float,
    seasonality: float,
    promo: int,
    past_features: np.ndarray = None # For a true TFT, this would be crucial
) -> int:
    """
    Predicts demand using a trained AdvancedDemandModel.
    For a production TFT, `past_features` would be required to build the sequence.
    Here, we simulate a sequence for prediction.

    Args:
        model (AdvancedDemandModel): The trained demand model for a specific SKU.
        price (float): The current price being considered.
        competitor_price (float): The competitor's price.
        seasonality (float): Current seasonality factor.
        promo (int): Promotional flag (0 or 1).
        past_features (np.ndarray): Optional. Array of past features for sequence.
                                     If None, a dummy sequence is created.

    Returns:
        int: The predicted demand, clamped at a minimum of 0.
    """
    # Create a dummy sequence for prediction if past_features not provided.
    # In a real-world scenario, you would feed the actual last SEQ_LEN days of data.
    if past_features is None:
        # Simulate a sequence of past data points
        # For simplicity, we'll use the current features repeated.
        # A more robust approach would fetch actual historical data.
        seq_features = np.array([[price, competitor_price, seasonality, promo]] * SEQ_LEN)
    else:
        # Ensure past_features has the correct shape (SEQ_LEN, 4)
        if past_features.shape != (SEQ_LEN, 4):
            raise ValueError(f"past_features must have shape ({SEQ_LEN}, 4), but got {past_features.shape}")
        seq_features = past_features

    # Convert to tensor and add batch dimension
    seq_tensor = torch.tensor(np.expand_dims(seq_features, axis=0), dtype=torch.float32)

    with torch.no_grad():
        prediction = model(seq_tensor).item()
    return max(0, int(prediction))