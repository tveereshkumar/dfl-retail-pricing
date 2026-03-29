import torch
import torch.nn as nn
import torch.nn.functional as F

class TFTDemandModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()

        # Static encoder
        self.static_layer = nn.Linear(input_size, hidden_size)

        # Temporal encoder
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # Attention-like gating
        self.attention = nn.Linear(hidden_size, hidden_size)

        # Output
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, time, features]

        static = torch.relu(self.static_layer(x))
        lstm_out, _ = self.lstm(static)

        attn = torch.sigmoid(self.attention(lstm_out))
        context = lstm_out * attn

        demand = self.fc(context[:, -1, :])

        return F.softplus(demand)