from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    DotProduct,
    ExpSineSquared,
    WhiteKernel,
)
from torchdiffeq import odeint


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasters.

    Notes
    -----
    Subclasses must implement :meth:`apply_forecast`.
    """

    @abstractmethod
    def apply_forecast(self, y_train, x_train, x_pred):
        """
        Fit the model on training data and predict on new data.

        Parameters
        ----------
        y_train : array-like
            Target values used to train the forecaster/regressor.
        x_train : array-like
            Feature matrix aligned with `y_train`. Content/shape depends on the
            specific forecaster implementation.
        x_pred : array-like
            Feature matrix for which predictions should be generated.

        Returns
        -------
        tuple
            A tuple ``(y_hat, y_hat_std)`` where:

            * **y_hat** : ndarray
                Point forecasts.
            * **y_hat_std** : ndarray or None
                Estimated standard deviation of each forecast if available;
                otherwise ``None``.
        """


class DirectForecaster(BaseForecaster):
    """
    Forecaster that uses a regressor directly for predictions.

    Parameters
    ----------
    regressor : object
        A scikit-learn compatible regressor implementing ``fit`` and ``predict``.
        The ``predict`` method is expected to optionally accept
        ``return_std=True`` and return a tuple ``(y_hat, y_hat_std)`` if
        supported.
    """

    def __init__(self, regressor):
        self.regressor = regressor

    def apply_forecast(self, y_train, x_train, x_pred):
        """Fit the regressor and make direct predictions."""
        self.regressor.fit(x_train, y_train)
        y_hat, y_hat_std = self.regressor.predict(x_pred, return_std=True)
        return y_hat, y_hat_std

class NCACellUpdate(nn.Module):
    """Local cellular update rule combining perception stencils and MLP adaptation."""

    def __init__(self, channels: int, hidden_dim: int = 128):
        super().__init__()
        self.channels = channels
        perception_channels = channels * 3
        
        self.net = nn.Sequential(
            nn.Conv2d(perception_channels, hidden_dim, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=(1, 1), bias=False)
        )
        nn.init.zeros_(self.net[-1].weight)

    def perceive(self, x):
        b, c, h, w = x.shape
        device = x.device
        sobel_x = torch.tensor([[-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 1, 3).repeat(c, 1, 1, 1)
        x_padded = F.pad(x, (1, 1, 0, 0), mode='replicate')
        grad_x = F.conv2d(x_padded, sobel_x, groups=c)
        grad_y = torch.zeros_like(x)
        return torch.cat([x, grad_x, grad_y], dim=1)

    def forward(self, x, mask=None):
        perceived = self.perceive(x)
        update = self.net(perceived)
        if mask is not None:
            update = update * mask
        return update


class NCAStatePool:
    """Checkpoint pool (Replay Buffer) for long-term NCA stability."""

    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self.pool = []

    def sample(self, batch_size: int, default_tensor: torch.Tensor) -> torch.Tensor:
        if len(self.pool) < batch_size:
            return default_tensor.clone()
        indices = [torch.randint(0, len(self.pool), (1,)).item() for _ in range(batch_size)]
        return torch.stack([self.pool[i] for i in indices], dim=0)

    def push(self, tensors: torch.Tensor):
        for tensor in tensors:
            if len(self.pool) >= self.capacity:
                self.pool.pop(torch.randint(0, len(self.pool), (1,)).item())
            self.pool.append(tensor.detach().clone())


class NCAForecaster(BaseForecaster):
    """Neural Cellular Automata time-series forecaster conforming to the BaseForecaster interface."""

    def __init__(
        self,
        hidden_dim: int = 64,
        update_probability: float = 0.5,
        seq_len: int = 12,
        epochs: int = 200,
        lr: float = 1e-3,
        overflow_weight: float = 100.0,
        pool_capacity: int = 512,
        device: str = None,
    ):
        self.hidden_dim = hidden_dim
        self.update_probability = update_probability
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        self.overflow_weight = overflow_weight
        self.pool_capacity = pool_capacity
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def apply_forecast(self, y_train, x_train, x_pred):
        y_train_arr = np.asarray(y_train, dtype=np.float32)
        if y_train_arr.ndim == 1:
            y_train_arr = y_train_arr[:, None]
            is_1d = True
        else:
            is_1d = False

        train_len = len(y_train_arr)
        pred_len = len(x_pred)
        feature_dim = y_train_arr.shape[1]

        actual_seq_len = min(self.seq_len, train_len - 1)
        if actual_seq_len < 2:
            actual_seq_len = max(1, train_len - 1)

        X_seq, Y_seq = [], []
        for i in range(train_len - actual_seq_len):
            X_seq.append(y_train_arr[i : i + actual_seq_len])
            Y_seq.append(y_train_arr[i + actual_seq_len])

        if len(X_seq) == 0:
            X_seq = [y_train_arr[:-1]]
            Y_seq = [y_train_arr[-1]]

        X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32, device=self.device).permute(0, 2, 1).unsqueeze(2)
        Y_tensor = torch.tensor(np.array(Y_seq), dtype=torch.float32, device=self.device)

        cell_update = NCACellUpdate(channels=feature_dim, hidden_dim=self.hidden_dim).to(self.device)
        state_pool = NCAStatePool(capacity=self.pool_capacity)

        optimizer = torch.optim.Adam(cell_update.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        cell_update.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            batch_size = X_tensor.shape[0]
            current_state = state_pool.sample(batch_size, X_tensor).to(self.device)

            delta = cell_update(current_state)
            if self.update_probability < 1.0:
                update_mask = (torch.rand(batch_size, 1, 1, actual_seq_len, device=self.device) < self.update_probability).float()
                delta = delta * update_mask

            next_state_grid = current_state + delta
            y_pred_train = next_state_grid[..., -1]

            loss_mse = criterion(y_pred_train, Y_tensor)
            loss_overflow = torch.mean(torch.abs(next_state_grid - torch.clamp(next_state_grid, -1.0, 1.0)))
            loss = loss_mse + self.overflow_weight * loss_overflow

            loss.backward()
            torch.nn.utils.clip_grad_norm_(cell_update.parameters(), max_norm=1.0)
            optimizer.step()

            state_pool.push(next_state_grid.detach())

        cell_update.eval()
        with torch.no_grad():
            current_seq = torch.tensor(
                y_train_arr[-actual_seq_len:][None, :, :],
                dtype=torch.float32,
                device=self.device
            ).permute(0, 2, 1).unsqueeze(2)

            forecasts = []
            for _ in range(pred_len):
                delta = cell_update(current_seq)
                current_seq = current_seq + delta
                next_val = current_seq[..., -1]
                forecasts.append(next_val.squeeze(2).permute(0, 2, 1).cpu().numpy())

                next_val_expanded = next_val.unsqueeze(3)
                current_seq = torch.cat([current_seq[..., 1:], next_val_expanded], dim=3)

            y_hat = np.concatenate(forecasts, axis=0)

        with torch.no_grad():
            y_hat_train_list = []
            for i in range(train_len):
                if i < actual_seq_len:
                    y_hat_train_list.append(y_train_arr[i])
                else:
                    seq_slice = torch.tensor(
                        y_train_arr[i - actual_seq_len:i][None, :, :],
                        dtype=torch.float32,
                        device=self.device
                    ).permute(0, 2, 1).unsqueeze(2)
                    delta = cell_update(seq_slice)
                    pred = (seq_slice + delta)[..., -1].squeeze(2).permute(0, 2, 1).cpu().numpy()[0]
                    y_hat_train_list.append(pred)
            y_train_pred_arr = np.array(y_hat_train_list)
            residual_std = np.std(y_train_arr - y_train_pred_arr, axis=0)

        y_hat_std = np.tile(residual_std, (pred_len, 1))

        if is_1d:
            y_hat = y_hat.flatten()
            y_hat_std = y_hat_std.flatten()

        return y_hat, y_hat_std


class SpectralConv1d(nn.Module):
    """1D Fourier Spectral Convolution layer for frequency-domain processing."""
    
    def __init__(self, in_channels: int, out_channels: int, num_modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_modes = num_modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, num_modes))
        self.weights_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, num_modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        
        # Compute 1D Fast Fourier Transform
        x_ft = torch.fft.rfft(x)

        # Initialize output Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1), dtype=torch.complex64, device=x.device)
        
        modes = min(self.num_modes, x_ft.size(-1))
        w = torch.complex(self.weights_real[..., :modes], self.weights_imag[..., :modes])
        
        # Multiply relevant Fourier modes
        out_ft[..., :modes] = torch.einsum("bix,iom->box", x_ft[..., :modes], w)

        # Inverse Fourier Transform back to temporal/spatial domain
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNO1dBlock(nn.Module):
    """Residual Fourier Neural Operator block with spectral convolution and skip connection."""

    def __init__(self, hidden_dim: int, num_modes: int):
        super().__init__()
        self.conv = SpectralConv1d(hidden_dim, hidden_dim, num_modes)
        self.w = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x) + self.w(x))


class FNOModel(nn.Module):
    """Full 1D FNO Architecture for sequence-to-next-step mapping."""

    def __init__(self, feature_dim: int, hidden_dim: int, num_modes: int, num_layers: int):
        super().__init__()
        self.lift = nn.Conv1d(feature_dim, hidden_dim, 1)
        self.fno_layers = nn.ModuleList([
            FNO1dBlock(hidden_dim, num_modes) for _ in range(num_layers)
        ])
        self.proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, feature_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, feature_dim) -> Transpose to (batch, feature_dim, seq_len)
        x = x.permute(0, 2, 1)
        x = self.lift(x)
        for layer in self.fno_layers:
            x = layer(x)
        x = self.proj(x)
        # Return the prediction for the final step in the sequence
        return x[..., -1]


class FNOForecaster(BaseForecaster):
    """1D Fourier Neural Operator time-series forecaster conforming to the BaseForecaster interface."""

    def __init__(
        self,
        hidden_dim: int = 32,
        num_modes: int = 8,
        num_layers: int = 2,
        seq_len: int = 12,
        epochs: int = 200,
        lr: float = 1e-3,
        device: str = None,
    ):
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def apply_forecast(self, y_train, x_train, x_pred):
        """Fit 1D FNO on training series and autoregressively forecast for x_pred steps."""
        y_train_arr = np.asarray(y_train, dtype=np.float32)
        
        if y_train_arr.ndim == 1:
            y_train_arr = y_train_arr[:, None]
            is_1d = True
        else:
            is_1d = False
            
        train_len = len(y_train_arr)
        pred_len = len(x_pred)
        feature_dim = y_train_arr.shape[1]

        actual_seq_len = min(self.seq_len, train_len - 1)
        if actual_seq_len < 2:
            actual_seq_len = max(1, train_len - 1)

        # Prepare sliding window tensors
        X_seq, Y_seq = [], []
        for i in range(train_len - actual_seq_len):
            X_seq.append(y_train_arr[i : i + actual_seq_len])
            Y_seq.append(y_train_arr[i + actual_seq_len])

        if len(X_seq) == 0:
            X_seq = [y_train_arr[:-1]]
            Y_seq = [y_train_arr[-1]]

        X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32, device=self.device)
        Y_tensor = torch.tensor(np.array(Y_seq), dtype=torch.float32, device=self.device)

        # Initialize FNO Model
        model = FNOModel(
            feature_dim=feature_dim,
            hidden_dim=self.hidden_dim,
            num_modes=self.num_modes,
            num_layers=self.num_layers
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # Training Loop
        model.train()
        print(f"--- [FNOForecaster] Starting Training ({self.epochs} Epochs) ---")
        log_interval = max(1, self.epochs // 10)
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            y_pred_train = model(X_tensor)
            loss = criterion(y_pred_train, Y_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % log_interval == 0 or epoch == 0 or (epoch + 1) == self.epochs:
                progress = ((epoch + 1) / self.epochs) * 100
                print(f"[FNO] Epoch [{epoch + 1:4d}/{self.epochs}] ({progress:5.1f}%) | MSE Loss: {loss.item():.6f}")

        # Autoregressive Forecasting Pass
        model.eval()
        with torch.no_grad():
            current_seq = torch.tensor(
                y_train_arr[-actual_seq_len:][None, :, :],
                dtype=torch.float32,
                device=self.device
            )

            forecasts = []
            for _ in range(pred_len):
                next_val = model(current_seq)  # shape: (1, feature_dim)
                forecasts.append(next_val.cpu().numpy())
                
                next_val_expanded = next_val.unsqueeze(1)
                current_seq = torch.cat([current_seq[:, 1:, :], next_val_expanded], dim=1)

            y_hat = np.concatenate(forecasts, axis=0)

        # Estimate residual uncertainty
        with torch.no_grad():
            y_hat_train_list = []
            for i in range(train_len):
                if i < actual_seq_len:
                    y_hat_train_list.append(y_train_arr[i])
                else:
                    seq_slice = torch.tensor(y_train_arr[i - actual_seq_len:i][None, :, :], dtype=torch.float32, device=self.device)
                    pred = model(seq_slice).cpu().numpy()[0]
                    y_hat_train_list.append(pred)
            y_train_pred_arr = np.array(y_hat_train_list)
            residual_std = np.std(y_train_arr - y_train_pred_arr, axis=0)

        y_hat_std = np.tile(residual_std, (pred_len, 1))

        if is_1d:
            y_hat = y_hat.flatten()
            y_hat_std = y_hat_std.flatten()

        return y_hat, y_hat_std


class DeepKoopmanModel(nn.Module):
    """
    Neural network module implementing the Deep Koopman architecture with 
    nonlinear encoder/decoder mappings and a constrained linear transition operator K.
    """
    def __init__(self, state_dim: int, lift_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.lift_dim = lift_dim
        
        # Nonlinear Observable Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, lift_dim)
        )
        
        # Nonlinear Observable Decoder
        self.decoder = nn.Sequential(
            nn.Linear(lift_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Linear transition operator K (lift_dim x lift_dim) 
        # Initialized close to a stable contraction mapping
        self.K = nn.Parameter(torch.eye(lift_dim) * 0.95 + torch.randn(lift_dim, lift_dim) * 0.01)

    def forward(self, x_history: torch.Tensor, pred_steps: int = 1) -> torch.Tensor:
        """
        Advances the system forward in the Koopman subspace and decodes back to physical space.
        Args:
            x_history: Tensor of shape (batch_size, seq_len, state_dim)
            pred_steps: Number of steps to forecast
        """
        x_last = x_history[:, -1, :]  # (batch_size, state_dim)
        g_current = self.encoder(x_last)  # (batch_size, lift_dim)
        
        predictions = []
        for _ in range(pred_steps):
            g_current = torch.matmul(g_current, self.K.T)
            x_pred = self.decoder(g_current)
            predictions.append(x_pred.unsqueeze(1))
            
        return torch.cat(predictions, dim=1)

    def compute_stability_loss(self) -> torch.Tensor:
        """
        Computes spectral radius penalty to guarantee ρ(K) <= 1.0, 
        preventing exponential explosion during multi-step rollouts.
        """
        eigvals = torch.linalg.eigvals(self.K)
        spectral_radius = torch.max(torch.abs(eigvals))
        return F.relu(spectral_radius - 1.0) ** 2


class DeepKoopmanForecaster(BaseForecaster):
    """Stable Deep Koopman Operator time-series forecaster conforming to the BaseForecaster interface."""

    def __init__(
        self,
        lift_dim: int = 64,
        hidden_dim: int = 64,
        epochs: int = 200,
        lr: float = 1e-3,
        alpha: float = 0.1,  # Weight for spectral stability regularization
        device: str = None,
    ):
        self.lift_dim = lift_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def apply_forecast(self, y_train, x_train, x_pred):
        """Fit Stable Deep Koopman operator on training data and forecast linearly in the lifted space."""
        y_train_arr = np.asarray(y_train, dtype=np.float32)
        
        if y_train_arr.ndim == 1:
            y_train_arr = y_train_arr[:, None]
            is_1d = True
        else:
            is_1d = False
            
        train_len = len(y_train_arr)
        pred_len = len(x_pred)
        feature_dim = y_train_arr.shape[1]

        # Initialize Stable Deep Koopman Model
        model = DeepKoopmanModel(
            state_dim=feature_dim,
            lift_dim=self.lift_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        y_tensor = torch.tensor(y_train_arr, dtype=torch.float32, device=self.device)

        # Training Loop: Jointly optimize reconstruction, linear dynamics, and spectral stability
        model.train()
        print(f"--- [DeepKoopmanForecaster] Starting Training ({self.epochs} Epochs) ---")
        log_interval = max(1, self.epochs // 10)
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Lift all time steps
            g = model.encoder(y_tensor)  # Shape: (train_len, lift_dim)
            
            # Predict next step in lifted space using linear operator K: g_{t+1} = g_t @ K.T
            g_next_pred = torch.matmul(g[:-1], model.K.T)
            g_next_true = g[1:]
            
            # Reconstruct original states
            y_reconstructed = model.decoder(g)
            
            # Combined Loss Components
            loss_recon = criterion(y_reconstructed, y_tensor)
            loss_dynamics = criterion(g_next_pred, g_next_true)
            loss_stability = model.compute_stability_loss()
            
            loss = loss_recon + loss_dynamics + (self.alpha * loss_stability)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if (epoch + 1) % log_interval == 0 or epoch == 0 or (epoch + 1) == self.epochs:
                progress = ((epoch + 1) / self.epochs) * 100
                print(
                    f"[DeepKoopman] Epoch [{epoch + 1:4d}/{self.epochs}] ({progress:5.1f}%) | "
                    f"Total Loss: {loss.item():.6f} | Recon: {loss_recon.item():.6f} | "
                    f"Dynamics: {loss_dynamics.item():.6f} | Stability: {loss_stability.item():.6f}"
                )

        # Forecasting Pass using stable multi-step rollout
        model.eval()
        with torch.no_grad():
            seq_len = min(10, train_len)
            x_history = y_tensor[-seq_len:].unsqueeze(0)  # Shape: (1, seq_len, feature_dim)
            y_hat_tensor = model(x_history, pred_steps=pred_len)  # Shape: (1, pred_len, feature_dim)
            y_hat = y_hat_tensor.squeeze(0).cpu().numpy()

        # Estimate residual standard deviation for variance/uncertainty output
        with torch.no_grad():
            g_train = model.encoder(y_tensor)
            y_train_pred = model.decoder(g_train).cpu().numpy()
            residual_std = np.std(y_train_arr - y_train_pred, axis=0)

        y_hat_std = np.tile(residual_std, (pred_len, 1))

        if is_1d:
            y_hat = y_hat.flatten()
            y_hat_std = y_hat_std.flatten()

        return y_hat, y_hat_std


class RNNForecaster(BaseForecaster):
    """Recurrent Neural Network (LSTM/GRU) time-series forecaster conforming to the BaseForecaster interface."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 1,
        cell_type: str = "lstm",
        seq_len: int = 10,
        epochs: int = 200,
        lr: float = 1e-3,
        device: str = None,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def apply_forecast(self, y_train, x_train, x_pred):
        """Fit RNN on training series and autoregressively forecast for x_pred steps."""
        y_train_arr = np.asarray(y_train, dtype=np.float32)
        
        if y_train_arr.ndim == 1:
            y_train_arr = y_train_arr[:, None]
            is_1d = True
        else:
            is_1d = False
            
        train_len = len(y_train_arr)
        pred_len = len(x_pred)
        feature_dim = y_train_arr.shape[1]

        # Ensure sequence length doesn't exceed available training data
        actual_seq_len = min(self.seq_len, train_len - 1)
        if actual_seq_len < 1:
            actual_seq_len = 1

        # Prepare sliding window training tensors
        X_seq, Y_seq = [], []
        for i in range(train_len - actual_seq_len):
            X_seq.append(y_train_arr[i : i + actual_seq_len])
            Y_seq.append(y_train_arr[i + actual_seq_len])

        if len(X_seq) == 0:
            X_seq = [y_train_arr[:-1]]
            Y_seq = [y_train_arr[-1]]

        X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32, device=self.device)
        Y_tensor = torch.tensor(np.array(Y_seq), dtype=torch.float32, device=self.device)

        # Build RNN Module
        if self.cell_type == "gru":
            self.rnn = nn.GRU(feature_dim, self.hidden_dim, self.num_layers, batch_first=True).to(self.device)
        else:
            self.rnn = nn.LSTM(feature_dim, self.hidden_dim, self.num_layers, batch_first=True).to(self.device)
            
        self.linear = nn.Linear(self.hidden_dim, feature_dim).to(self.device)

        optimizer = torch.optim.Adam(
            list(self.rnn.parameters()) + list(self.linear.parameters()),
            lr=self.lr,
        )
        criterion = nn.MSELoss()

        # Training loop
        self.rnn.train()
        print(f"--- [RNNForecaster ({self.cell_type.upper()})] Starting Training ({self.epochs} Epochs) ---")
        log_interval = max(1, self.epochs // 10)
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            out, _ = self.rnn(X_tensor)
            last_out = out[:, -1, :]
            y_pred_train = self.linear(last_out)
            loss = criterion(y_pred_train, Y_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % log_interval == 0 or epoch == 0 or (epoch + 1) == self.epochs:
                progress = ((epoch + 1) / self.epochs) * 100
                print(f"[RNN ({self.cell_type.upper()})] Epoch [{epoch + 1:4d}/{self.epochs}] ({progress:5.1f}%) | MSE Loss: {loss.item():.6f}")

        # Autoregressive Forecasting for pred_len steps
        self.rnn.eval()
        with torch.no_grad():
            current_seq = torch.tensor(
                y_train_arr[-actual_seq_len:][None, :, :],
                dtype=torch.float32,
                device=self.device
            )

            forecasts = []
            for _ in range(pred_len):
                out, _ = self.rnn(current_seq)
                next_val = self.linear(out[:, -1, :])
                forecasts.append(next_val.cpu().numpy())
                
                next_val_expanded = next_val.unsqueeze(1)
                current_seq = torch.cat([current_seq[:, 1:, :], next_val_expanded], dim=1)

            y_hat = np.concatenate(forecasts, axis=0)

        # In-sample reconstruction for residual error estimation
        with torch.no_grad():
            y_hat_train_list = []
            for i in range(train_len):
                if i < actual_seq_len:
                    y_hat_train_list.append(y_train_arr[i])
                else:
                    seq_slice = torch.tensor(y_train_arr[i - actual_seq_len:i][None, :, :], dtype=torch.float32, device=self.device)
                    out, _ = self.rnn(seq_slice)
                    pred = self.linear(out[:, -1, :]).cpu().numpy()[0]
                    y_hat_train_list.append(pred)
            y_train_pred_arr = np.array(y_hat_train_list)
            residual_std = np.std(y_train_arr - y_train_pred_arr, axis=0)

        y_hat_std = np.tile(residual_std, (pred_len, 1))

        if is_1d:
            y_hat = y_hat.flatten()
            y_hat_std = y_hat_std.flatten()

        return y_hat, y_hat_std


class ODEVectorField(nn.Module):
    """Parameterizes the neural vector field dh/dt = f(t, h ; theta)."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        # Inputs: hidden state h + scalar time t
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # Robustly broadcast scalar/0D tensor time 't' across batch dimension
        t_vec = torch.full((h.shape[0], 1), fill_value=t.item(), device=h.device, dtype=h.dtype)
        return self.net(torch.cat([h, t_vec], dim=-1))


class NeuralODEForecaster(BaseForecaster):
    """Neural ODE time-series forecaster conforming to the BaseForecaster interface."""

    def __init__(
        self,
        hidden_dim: int = 32,
        epochs: int = 250,
        lr: float = 1e-2,
        solver: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
        device: str = None,
    ):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def apply_forecast(self, y_train, x_train, x_pred):
        """Fit Neural ODE on training series and forecast for x_pred steps."""
        y_train_arr = np.asarray(y_train, dtype=np.float32)
        
        # Preserve 1D vs 2D structure (e.g., latent components)
        if y_train_arr.ndim == 1:
            y_train_arr = y_train_arr[:, None]  # Shape: (T, 1)
            is_1d = True
        else:
            is_1d = False
            
        train_len = len(x_train)
        pred_len = len(x_pred)
        feature_dim = y_train_arr.shape[1]

        # 1. Independent Time Grid: Train on [0, 1], forecast past t = 1.0
        dt = 1.0 / max(train_len - 1, 1)
        t_train_grid = np.linspace(0.0, 1.0, train_len, dtype=np.float32)
        t_pred_grid = np.linspace(1.0 + dt, 1.0 + dt * pred_len, pred_len, dtype=np.float32)
        
        t_train = torch.tensor(t_train_grid, device=self.device)
        t_full = torch.tensor(np.concatenate([t_train_grid, t_pred_grid]), device=self.device)
        
        y_train_t = torch.tensor(y_train_arr, dtype=torch.float32, device=self.device)

        # Architectural Encoder / Decoder Modules
        encoder = nn.Linear(feature_dim, self.hidden_dim).to(self.device)
        func = ODEVectorField(hidden_dim=self.hidden_dim).to(self.device)
        readout = nn.Linear(self.hidden_dim, feature_dim).to(self.device)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(func.parameters()) + list(readout.parameters()),
            lr=self.lr,
        )
        criterion = nn.MSELoss()

        # Initial condition h0 from first observed target state y(0)
        y0_input = torch.tensor(y_train_arr[0:1], dtype=torch.float32, device=self.device)  # Shape: (1, feature_dim)

        # 2. Training Loop
        print(f"--- [NeuralODEForecaster] Starting Training ({self.epochs} Epochs) ---")
        log_interval = max(1, self.epochs // 10)
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            h0 = encoder(y0_input)  # Shape: (1, hidden_dim)
            
            # Solve ODE trajectory across training horizon
            h_traj = odeint(
                func,
                h0,
                t_train,
                method=self.solver,
                rtol=self.rtol,
                atol=self.atol,
            )  # Output shape: (train_len, 1, hidden_dim)

            y_hat_train = readout(h_traj.squeeze(1))  # Shape: (train_len, feature_dim)
            loss = criterion(y_hat_train, y_train_t)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % log_interval == 0 or epoch == 0 or (epoch + 1) == self.epochs:
                progress = ((epoch + 1) / self.epochs) * 100
                print(f"[NeuralODE] Epoch [{epoch + 1:4d}/{self.epochs}] ({progress:5.1f}%) | MSE Loss: {loss.item():.6f}")

        # 3. Forecast Pass
        with torch.no_grad():
            h0 = encoder(y0_input)
            h_full = odeint(
                func,
                h0,
                t_full,
                method=self.solver,
                rtol=self.rtol,
                atol=self.atol,
            )
            y_hat_full = readout(h_full.squeeze(1)).cpu().numpy()
            y_hat = y_hat_full[train_len:]  # Extract out-of-sample steps

        # Estimate residual standard deviation for variance/uncertainty output
        residual_std = np.std(y_train_arr - y_hat_full[:train_len].reshape(y_train_arr.shape), axis=0)
        y_hat_std = np.tile(residual_std, (pred_len, 1))

        if is_1d:
            y_hat = y_hat.flatten()
            y_hat_std = y_hat_std.flatten()

        return y_hat, y_hat_std


class RecursiveForecaster(BaseForecaster):
    """Forecaster that uses a recursive strategy for multi-step predictions."""

    def __init__(self, regressor, lags=10, window_size=10):
        self.regressor = regressor
        self.lags = lags
        self.window_size = window_size

    def apply_forecast(self, y_train, _x_train, x_pred):
        """Fit a recursive forecaster using the provided regressor."""
        window_features = RollingFeatures(
            stats=["mean"], window_sizes=self.window_size
        )

        forecaster = ForecasterRecursive(
            regressor=self.regressor,
            lags=self.lags,
            window_features=window_features,
        )

        forecaster.fit(pd.Series(y_train))
        steps = x_pred.shape[0]
        y_hat = forecaster.predict(steps=steps)
        y_hat_std = y_hat

        return y_hat, y_hat_std


class RecursiveForecaster(BaseForecaster):
    """Forecaster that uses a recursive strategy for multi-step predictions."""

    def __init__(self, regressor, lags=10, window_size=10):
        self.regressor = regressor
        self.lags = lags
        self.window_size = window_size

    def apply_forecast(self, y_train, _x_train, x_pred):
        """Fit a recursive forecaster using the provided regressor."""
        window_features = RollingFeatures(
            stats=["mean"], window_sizes=self.window_size
        )

        forecaster = ForecasterRecursive(
            regressor=self.regressor,
            lags=self.lags,
            window_features=window_features,
        )

        forecaster.fit(pd.Series(y_train))
        steps = x_pred.shape[0]
        y_hat = forecaster.predict(steps=steps)
        y_hat_std = y_hat

        return y_hat, y_hat_std


def create_gp_regressor(
    data_length=45,
    trend_scale=0.1,
    irregularities_scale=10.0,
    noise_scale=2.0,
    optimize_restarts=8,
    random_seed=42,
):
    """Create a Gaussian Process regressor with basic parameter configuration."""
    scale_factor = 5.0 / data_length
    trend_kernel = trend_scale * DotProduct(sigma_0=0.0)
    irregularities_kernel = irregularities_scale * ExpSineSquared(
        length_scale=scale_factor, periodicity=scale_factor
    )
    noise_kernel = noise_scale * WhiteKernel(noise_level=1.0)
    kernel = irregularities_kernel + noise_kernel + trend_kernel

    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=optimize_restarts,
        random_state=random_seed,
    )


# --- Instantiations ---
gp_recursive_forecaster = RecursiveForecaster(
    create_gp_regressor(), lags=10, window_size=10
)
gp_forecaster = DirectForecaster(create_gp_regressor())
neural_ode_forecaster = NeuralODEForecaster()
rnn_forecaster = RNNForecaster(cell_type="lstm", seq_len=10, epochs=200)
deep_koopman_forecaster = DeepKoopmanForecaster(lift_dim=64, epochs=200)
fno_forecaster = FNOForecaster(hidden_dim=32, num_modes=8, num_layers=2, seq_len=12, epochs=200)
nca_forecaster = NCAForecaster(hidden_dim=64, seq_len=12, epochs=200)
# --- Forecaster Registry ---
forecast_techniques = {
    "GaussianProcessForecaster": gp_forecaster,
    "GaussianProcessRecursiveForecaster": gp_recursive_forecaster,
    "NeuralODEForecaster": neural_ode_forecaster,
    "RNNForecaster": rnn_forecaster,
    "FNOForecaster": fno_forecaster,
    "DeepKoopmanForecaster": deep_koopman_forecaster,
    "NCAForecaster": nca_forecaster
}