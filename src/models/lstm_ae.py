"""
SW-ADP Stage 4: LSTM Autoencoder
DL baseline for spacecraft anomaly detection (Hundman et al. KDD 2018 paradigm).
Trained on normal segments only; reconstruction error used as anomaly score.
"""
import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        _, (h_n, c_n) = self.lstm(x)
        # h_n: [n_layers, batch, hidden_dim] -- use last layer
        return h_n, c_n


class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h_0, c_0):
        # x: [batch, seq_len, input_dim] (repeated bottleneck or teacher-forced)
        out, _ = self.lstm(x, (h_0, c_0))
        # out: [batch, seq_len, hidden_dim]
        return self.fc(out)  # [batch, seq_len, output_dim]


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for time-series reconstruction.

    Encoder compresses the sequence into a fixed-size hidden state.
    Decoder reconstructs the sequence from this bottleneck.
    Reconstruction error = anomaly score.
    """
    def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.encoder = LSTMEncoder(input_dim, hidden_dim, n_layers, dropout)
        # Decoder input: repeat last encoder hidden state at each timestep
        self.decoder = LSTMDecoder(hidden_dim, hidden_dim, input_dim, n_layers, dropout)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            reconstructed: [batch, seq_len, input_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Encode
        h_n, c_n = self.encoder(x)

        # Create decoder input: repeat last hidden state across all timesteps
        # h_n[-1] is the last layer's hidden state: [batch, hidden_dim]
        bottleneck = h_n[-1].unsqueeze(1).repeat(1, seq_len, 1)  # [batch, seq_len, hidden_dim]

        # Decode
        reconstructed = self.decoder(bottleneck, h_n, c_n)
        return reconstructed

    def get_model_size(self):
        """Return model size in parameters and approximate KB."""
        n_params = sum(p.numel() for p in self.parameters())
        size_kb = n_params * 4 / 1024  # float32
        return n_params, size_kb
