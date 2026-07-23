from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import (
    PCA,
    KernelPCA,
)


class DimensionalityReduction(ABC):
    """Abstract interface for dimensionality reduction techniques.

    Subclasses must implement the core API used throughout the code base.
    """

    @abstractmethod
    def decompose(self):
        """Fit the model and project the data into a lower-dimensional space.

        This method should set any attributes necessary for later reconstruction.
        """

    @staticmethod
    @abstractmethod
    def reconstruct_predictions():
        """Reconstruct data from previously predicted component scores.

        Implementations should not rely on instance state since the method is static.
        """

    @abstractmethod
    def reconstruct_components(self):
        """Reconstruct the original space using the stored component scores."""

    @abstractmethod
    def get_component(self):
        """Return a spatial map corresponding to a single component."""

    @abstractmethod
    def error(self):
        """Compute an error metric (e.g., RMSE) between reconstructions and truth."""

    @abstractmethod
    def set_from_simulation(self):
        """Attach metadata (shape, scaling, etc.) from a Simulation object."""

# ============= PyTorch Convolutional Autoencoder Network Architecture =============
class ConvVAE(nn.Module):
    """
    A foundational Convolutional Variational Autoencoder layout 
    tailored dynamically for structural spatial grid fields.
    """
    def __init__(self, latent_dim: int, in_channels: int = 1, in_shape: tuple = (64, 64)):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.in_shape = in_shape
        
        # 1. Encoder network block
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Track the exact dynamic output shape of the convolutional layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *in_shape)
            conv_features = self.encoder_conv(dummy_input)
            _, self.hidden_c, self.hidden_h, self.hidden_w = conv_features.shape
            flat_features = conv_features.numel() // conv_features.shape[0]
            
        self.fc_mu = nn.Linear(flat_features, latent_dim)
        self.fc_logvar = nn.Linear(flat_features, latent_dim)
        
        # 2. Decoder network block
        self.fc_decode = nn.Linear(latent_dim, flat_features)
        
        # Dynamically set target sizes to guarantee multiplication sanity check matches perfectly
        self.unflatten_shape = (self.hidden_c, self.hidden_h, self.hidden_w)
        
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=self.unflatten_shape)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        features = self.encoder_conv(x)
        flat_enc = features.view(features.size(0), -1)
        
        mu = self.fc_mu(flat_enc)
        logvar = self.fc_logvar(flat_enc)
        z = self.reparameterize(mu, logvar)
        
        decoded_flat = self.fc_decode(z)
        reconstructed = self.decoder_conv(self.unflatten(decoded_flat))
        
        # Optional: crop/pad output if ConvTranspose introduces dimensional rounding drift vs raw input
        if reconstructed.shape[-2:] != x.shape[-2:]:
            reconstructed = nn.functional.interpolate(reconstructed, size=x.shape[-2:], mode='bilinear', align_corners=False)
            
        return reconstructed, mu, logvar


# ============= Dimensionality Reduction CVAE Framework =============
class DimensionalityReductionCVAE:
    """Dimensionality Reduction using a Convolutional Variational Autoencoder"""
    def __init__(self, comp: int, epochs: int = 40, batch_size: int = 16, lr: float = 1e-3, beta: float = 0.01):
        self.components = None
        self.comp = comp  # Latent dimensions
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta = beta  # Weighting parameter for KL Divergence
        self.model = None 
        self.shape = None # Expects (Depth, Height, Width) or (Height, Width)
        self.bool_mask = None
        self.int_mask = None
        self.desc = None
        self.simulation = None
        self.time_dim = None
        self.len = None
        print(f"Using Convolutional VAE (Latent components: {comp}, Beta: {beta})")

    def set_from_simulation(self, sim):
        """Copy metadata framework directly from a Simulation instance."""
        self.time_dim = getattr(sim, 'time_dim', None)
        self.shape = getattr(sim, 'shape', None)
        self.len = getattr(sim, 'len', None)
        self.desc = getattr(sim, 'desc', None)
        self.simulation = getattr(sim, 'simulation', None)

    def vae_loss_function(self, recon_x, x, mu, logvar):
        """Computes reconstruction loss (MSE) and balances it against 
        the Kullback-Leibler Divergence regularizer."""
        # Fixed capitalization from nn.Functional to nn.functional
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
        kl_element = 1 + logvar - mu.pow(2) - logvar.exp()
        kl_loss = -0.5 * torch.mean(torch.sum(kl_element, dim=1))
        total_loss = recon_loss + (self.beta * kl_loss)
        return total_loss, recon_loss, kl_loss

    def decompose(self, simulation_array, length, info_desc=None):
        """Train the VAE on the historical sequence and extract structural components."""
        array = np.copy(simulation_array[:length])
        
        # Capture 3D structural fields: (Depth_Channels, Y, X)
        self.shape = array.shape[1:] 
        in_channels = self.shape[0] if len(self.shape) == 3 else 1
        spatial_shape = self.shape[-2:]
        
        # Generate structural land-mask validation maps
        self.bool_mask = np.isfinite(array[0])
        array[np.isnan(array)] = 0.0
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvVAE(latent_dim=self.comp, in_channels=in_channels, in_shape=spatial_shape).to(device)
        
        tensor_data = torch.tensor(array, dtype=torch.float32)
        if tensor_data.dim() == 3:  # Fallback encapsulation if data lacks a depth/channel dimension
            tensor_data = tensor_data.unsqueeze(1)
            
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            
            for batch in dataloader:
                inputs = batch[0].to(device)
                optimizer.zero_grad()
                
                outputs, mu, logvar = self.model(inputs)
                loss, r_loss, kl = self.vae_loss_function(outputs, inputs, mu, logvar)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * inputs.size(0)
                epoch_recon += r_loss.item() * inputs.size(0)
                epoch_kl += kl.item() * inputs.size(0)
                
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:02d}/{self.epochs} | Total Loss: {epoch_loss/length:.6f} "
                      f"(Recon MSE: {epoch_recon/length:.6f}, KL Div: {epoch_kl/length:.6f})")
        
        self.model.eval()
        with torch.no_grad():
            full_tensor = torch.tensor(array, dtype=torch.float32).to(device)
            if full_tensor.dim() == 3:
                full_tensor = full_tensor.unsqueeze(1)
            _, mu_components, _ = self.model(full_tensor)
            
        self.components = mu_components.cpu().numpy()
        return self.components, self.model, self.bool_mask

    def reconstruct_predictions(self, predictions, n, info, begin=0):
        """Pass-through configuration to route baseline pipeline calls to CVAE space."""
        # Cleanly convert incoming prediction matrix to NumPy array whether it arrives as a DataFrame or Array
        if isinstance(predictions, pd.DataFrame):
            forecast_scores = predictions.iloc[begin:, :n].to_numpy()
        else:
            forecast_scores = np.asarray(predictions)[begin:, :n]
            
        # Reconstruct fields from the sliced forecast scores Matrix
        reconstructed_data = self.reconstruct_components(n=n, custom_scores=forecast_scores)
        
        # Build standard int_mask pattern to remain consistent with simulation ecosystem expects
        mask_source = info["mask"] if (info is not None and "mask" in info) else self.bool_mask
        int_mask = mask_source.astype(np.int32).reshape(self.shape)
        
        # Extract scaling statistics securely
        desc_stats = info["desc"] if (info is not None and "desc" in info) else self.desc
        
        if desc_stats and "std" in desc_stats:
            reconstructed_data = reconstructed_data * 2 * desc_stats["std"] + desc_stats["mean"]
            
        return int_mask, reconstructed_data

    def reconstruct_components(self, n: int, custom_scores=None):
        """Pass structural parameters back through the decoder to evaluate structural tracking."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Fall back to internal trained components if custom forecast scores are not supplied
        if custom_scores is not None:
            latent_scores = np.copy(custom_scores)
        else:
            latent_scores = np.copy(self.components[:, :n])
            
        # If the number of components requested is less than model capacity, pad with zeros
        if n < self.comp:
            padding = np.zeros((len(latent_scores), self.comp - n))
            latent_scores = np.hstack([latent_scores, padding])
            
        latent_tensor = torch.tensor(latent_scores, dtype=torch.float32).to(device)
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model.decoder_conv(self.model.unflatten(self.model.fc_decode(latent_tensor)))
            reconstructed = reconstructed.cpu().numpy()
            
        # 1. Remove the singleton channel dimension if present from the decoder output
        if reconstructed.shape[1] == 1:
            reconstructed = np.squeeze(reconstructed, axis=1)
            
        # 2. Extract target spatial dimensions from the domain mask
        if len(self.bool_mask.shape) == 3:
            _, target_lat, target_lon = self.bool_mask.shape
        else:
            target_lat, target_lon = self.bool_mask.shape[-2:]
            
        # 3. Slice out convolutional layer padding to align dimensions (e.g., 52 -> 50)
        if reconstructed.shape[-2] != target_lat or reconstructed.shape[-1] != target_lon:
            reconstructed = reconstructed[..., :target_lat, :target_lon]
            
        # 4. Ensure the mask has the correct dimensions for robust time-series broadcasting
        mask_expanded = self.bool_mask[None, ...] if len(self.bool_mask.shape) == 2 else self.bool_mask
        if len(reconstructed.shape) > len(mask_expanded.shape):
            mask_expanded = mask_expanded[None, ...]
            
        # 5. Foolproof broadcasting: wherever the mask is False, force NaN.
        reconstructed = np.where(mask_expanded, reconstructed, np.nan)
            
        return reconstructed

    def get_component(self, n):
        """Return a pseudo spatial map representation generated from a base latent unit spike."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        latent_vector = np.zeros((1, self.comp))
        if n < self.comp:
            latent_vector[0, n] = 1.0 
            
        latent_tensor = torch.tensor(latent_vector, dtype=torch.float32).to(device)
        self.model.eval()
        with torch.no_grad():
            component_map = self.model.decoder_conv(self.model.unflatten(self.model.fc_decode(latent_tensor)))
            component_map = component_map.cpu().numpy()[0]
            
        if component_map.shape[0] == 1:
            component_map = np.squeeze(component_map, axis=0)
            
        component_map[~self.bool_mask] = np.nan
        if self.desc and "std" in self.desc:
            component_map = component_map * 2 * self.desc["std"] + self.desc["mean"]
        return component_map

    def error(self, n):
        """Compute error metric (RMSE) between reconstructions and truth in physical scale."""
        reconstruction = self.reconstruct_components(n)
        truth = np.copy(self.simulation)
        
        if truth is None:
            raise ValueError("Base target simulation data matrix is missing or unassigned.")
            
            # Undo standardization on BOTH reconstruction AND truth so scales match
            # 
        if self.desc and "std" in self.desc:
            reconstruction = reconstruction * 2 * self.desc["std"] + self.desc["mean"]
        if hasattr(truth, 'values'):
            truth = truth.values
        
        truth = truth * 2 * self.desc["std"] + self.desc["mean"]
        
        if len(np.shape(truth)) == 3:
            valid_count = np.count_nonzero(~np.isnan(truth[0]))
            rmse_values = np.sqrt(np.nansum((truth - reconstruction) ** 2, axis=(1, 2)) / valid_count)
            rmse_map = np.sqrt(np.nansum((truth - reconstruction) ** 2, axis=0) / self.len)
        
        else:
            valid_count = np.count_nonzero(~np.isnan(truth[0]), axis=(1, 2))
            rmse_values = np.nansum((truth - reconstruction) ** 2, axis=(2, 3))
            
            for i in range(len(valid_count)):
                rmse_values[:, i] = rmse_values[:, i] / valid_count[i]
            rmse_values = np.sqrt(rmse_values)
            rmse_map = np.sqrt(np.sum((truth - reconstruction) ** 2, axis=0) / self.len)
        
        return reconstruction, rmse_values, rmse_map

    def save_weights(self, base_path: str):
        """Save the PyTorch state dict securely to its dedicated weight file."""
        if os.path.isdir(base_path):
             weight_path = os.path.join(base_path, "vae_weights.pt")
        else:
            clean_path = base_path.replace(".npz", "").replace("_weights.pt", "").replace(".pkl", "")
            dirname, basename = os.path.split(clean_path)
            if basename.startswith("pca_"):
                basename = basename[4:]
            weight_path = os.path.join(dirname, f"{basename}_vae_weights.pt")
        
        if self.model is not None:
            torch.save(self.model.state_dict(), weight_path)
            print(f"[CVAE] Model weights successfully written to: {weight_path}")

    def load_weights(self, base_path: str, in_channels: int = None, spatial_shape: tuple = None):
        """Re-instantiates the network block and maps the extracted state dictionary."""
        if in_channels is None:
            in_channels = self.shape[0] if (self.shape and len(self.shape) == 3) else 1
        if spatial_shape is None:
            spatial_shape = self.shape[-2:] if self.shape else (64, 64)
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvVAE(latent_dim=self.comp, in_channels=in_channels, in_shape=spatial_shape).to(device)
        
        if os.path.isdir(base_path):
            weight_path = os.path.join(base_path, "vae_weights.pt")
        else:
            clean_path = base_path.replace(".npz", "").replace("_weights.pt", "").replace(".pkl", "")
            dirname, basename = os.path.split(clean_path)
            if basename.startswith("pca_"):
                basename = basename[4:]
            weight_path = os.path.join(dirname, f"{basename}_vae_weights.pt")
        try:
            self.model.load_state_dict(torch.load(weight_path, map_location=device))
            self.model.eval()
            print(f"[CVAE] Network weights successfully reloaded from: {weight_path}")
        except FileNotFoundError:
            alt_path = weight_path.replace(basename, f"pca_{basename}")
            if os.path.exists(alt_path):
                self.model.load_state_dict(torch.load(alt_path, map_location=device))
                self.model.eval()
                print(f"[CVAE] Network weights successfully reloaded from fallback: {alt_path}")
            else:
                print(f"[CVAE] Warning: Weight file not found at {weight_path}. Model remains un-initialized.")

    def __getstate__(self):
        """Prepares class state for pickle by stripping out un-picklable PyTorch model objects."""
        state = self.__dict__.copy()
        state['model'] = None
        return state

    def __setstate__(self, state):
        """Restores class state after pickle extraction."""
        self.__dict__.update(state)
        self.model = None
 
class DimensionalityReductionPCA(DimensionalityReduction):
    """Dimensionality reduction using classical Principal Component Analysis (PCA).

    Parameters
    ----------
    comp : int
        Number of principal components to retain.

    Attributes
    ----------
    components : ndarray or None
        Projected data of shape ``(time, comp)`` after :meth:`decompose`.
    comp : int
        Requested number of components.
    pca : sklearn.decomposition.PCA or None
        Fitted PCA object.
    shape : tuple[int, ...] or None
        Original spatial shape of a single time slice (e.g. ``(height, width)``).
    desc : dict or None
        Dictionary with keys such as ``'mean'`` and ``'std'`` used for rescaling.
    bool_mask : ndarray of bool or None
        Mask of valid (finite) features in flattened space.
    time_dim, len, simulation : various
        Metadata copied from the provided Simulation object
        via :meth:`set_from_simulation`.
    """

    def __init__(self, comp):
        self.components = None
        self.comp = comp
        self.pca = None
        self.shape = None
        self.desc = None
        print("Using normal PCA")

    def set_from_simulation(self, sim):
        """Copy metadata from a ``Simulation`` instance.

        Parameters
        ----------
        sim : object
            Object possessing attributes ``time_dim``, ``shape``, ``len``, ``desc`` and
            ``simulation`` (NumPy array). No type enforcement is done here.
        """
        self.time_dim = sim.time_dim
        self.shape = sim.shape
        self.len = sim.len
        self.desc = sim.desc
        self.simulation = sim.simulation

    def get_num_components(self):
        """Return the number of components retained by the fitted PCA.

        Returns
        -------
        int
            Number of components learned (``pca.n_components_``).
        """
        return self.pca.n_components_

    def decompose(self, simulation, length):
        """Run PCA on ``simulation``.

        Parameters
        ----------
        simulation : ndarray
            Shape ``(time, *spatial_dims)``.
        length : int
            Number of time steps to use.

        Returns
        -------
        components : ndarray
            Shape ``(length, comp)``.
        pca : PCA
            Fitted PCA object.
        bool_mask : ndarray[bool]
            Mask of finite features in the flattened space.
        """
        array = simulation.reshape(length, -1)

        self.bool_mask = np.asarray(np.isfinite(array[0, :]), dtype=bool)
        array_masked = array[:, self.bool_mask]
        pca = PCA(self.comp, whiten=False)
        self.components = pca.fit_transform(array_masked)
        self.pca = pca

        return self.components, self.pca, self.bool_mask

    @staticmethod
    def reconstruct_predictions(predictions, n, info, begin=0):
        """Rebuild fields from predicted PCA scores.

        Parameters
        ----------
        predictions : pandas.DataFrame
            Rows = time, columns = components.
        n : int
            Number of components to use.
        info : dict
            Keys: ``'mask'``, ``'shape'``, ``'pca'``, ``'desc'``.
        begin : int, default 0
            First row to reconstruct.

        Returns
        -------
        int_mask : ndarray[int]
            ``info['mask']`` reshaped.
        rec : ndarray
            Reconstructed array, rescaled.
        """
        rec = []
        int_mask = info["mask"].astype(np.int32).reshape(info["shape"])
        # Reconstruct each year/time interval from components
        for t in range(begin, len(predictions)):
            map_ = np.zeros((info["shape"]), dtype=np.float32)
            arr = np.array(
                list(predictions.iloc[t, :n]) + [0] * (len(info["pca"].components_) - n)
            )

            # Reshape arr to 2D (sample, n_components) for inverse_transform,
            # flatten inverse transform output, assign output to orginal ocean grid
            map_[int_mask == 1] = (
                info["pca"].inverse_transform(arr.reshape(1, -1)).flatten()
            )

            map_[int_mask == 0] = np.nan
            rec.append(map_)
        return int_mask, np.array(rec) * 2 * info["desc"]["std"] + info["desc"]["mean"]

    def reconstruct_components(self, n):
        """
        Reconstruct data using a specified number of principal components.

        Parameters
        ----------
        n : int
            The number of components used for reconstruction.

        Returns
        -------
        numpy.ndarray
            The reconstructed data.
        """
        rec = []
        # int_mask =   # Convert the boolean mask to int mask once
        self.int_mask = self.bool_mask.astype(np.int32).reshape(
            self.shape
        )  # Reshape to match the shape of map_

        # Reconstruct each year/time interval from components
        for t in range(len(self.components)):
            map_ = np.zeros(self.shape, dtype=np.float32)
            arr = np.array(
                list(self.components[t, :n]) + [0] * (len(self.pca.components_) - n)
            )
            # Reshape arr to 2D (sample, n_components) for inverse_transform,
            # flatten inverse transform output, assign output to orginal ocean grid
            map_[self.int_mask == 1] = self.pca.inverse_transform(
                arr.reshape(1, -1)
            ).flatten()
            map_[self.int_mask == 0] = np.nan
            rec.append(map_)
        return np.array(rec)

    def get_component(self, n):
        """
        Get an approximate kernel principal component map for component n.

        For non-linear kernels the mapping is implicit, so this is only a proxy.

        Parameters
        ----------
        n : int
            Component index.

        Returns
        -------
        ndarray
            ``self.shape``: A 2D map corresponding to the n-th component.
        """
        # map_ = np.zeros((np.product(self.shape)), dtype=float)
        map_ = np.zeros((np.prod(self.shape)), dtype=float)
        map_[~self.bool_mask] = np.nan
        map_[self.bool_mask] = self.pca.components_[n]
        map_ = map_.reshape(self.shape)

        map_ = 2 * map_ * self.desc["std"] + self.desc["mean"]
        return map_

    def error(self, n):
        """Alias of :meth:`rmse`.

        Parameters
        ----------
        n : int
            Components used in reconstruction.
        """
        return self.rmse(n)

    def rmse(self, n):
        """RMSE using ``n`` components.

        Parameters
        ----------
        n : int
            Components used in reconstruction.

        Returns
        -------
        reconstruction : ndarray
        rmse_values : ndarray
            Per-time RMSE.
        rmse_map : ndarray
            Per-point RMSE.
        """
        reconstruction = self.reconstruct_components(n)
        rmse_values = self.rmse_values(reconstruction) * 2 * self.desc["std"]
        rmse_map = self.rmse_map(reconstruction) * 2 * self.desc["std"]
        return reconstruction, rmse_values, rmse_map

    def rmse_values(self, reconstruction):
        """RMSE per time sample.

        Parameters
        ----------
        reconstruction : ndarray
            Output of :meth:`reconstruct_components`.

        Returns
        -------
        ndarray
            RMSE for each time index.
        """
        truth = self.simulation  # * 2 * self.desc["std"] + self.desc["mean"]
        rec = reconstruction  # * 2 * self.desc["std"] + self.desc["mean"]
        if len(np.shape(truth)) == 3:
            n = np.count_nonzero(~np.isnan(truth[0]))

            rmse_values = np.sqrt(np.nansum((truth - rec) ** 2, axis=(1, 2)) / n)
        else:
            n = np.count_nonzero(~np.isnan(self.simulation[0]), axis=(1, 2))

            rmse_values = np.nansum((truth - rec) ** 2, axis=(2, 3))
            for i in range(len(n)):
                rmse_values[:, i] = rmse_values[:, i] / n[i]
            rmse_values = np.sqrt(rmse_values)
        return rmse_values

    def rmse_map(self, reconstruction):
        """RMSE per spatial location.

        Parameters
        ----------
        reconstruction : ndarray
            Output of :meth:`reconstruct_components`.

        Returns
        -------
        ndarray
            ``self.shape`` RMSE map.
        """
        t = self.len
        return np.sqrt(np.sum((self.simulation[:] - reconstruction) ** 2, axis=0) / t)


class DimensionalityReductionKernelPCA(DimensionalityReduction):
    """Kernel PCA-based reduction.

    Parameters
    ----------
    comp : int
        Components to keep.
    kernel : str, default 'rbf'
        Kernel passed to :class:`KernelPCA`.
    **kwargs
        Extra ``KernelPCA`` args.

    Attributes
    ----------
    components : ndarray | None
        Projected data, ``(time, comp)``.
    pca : KernelPCA | None
        Fitted KernelPCA estimator.
    shape : tuple[int, ...] | None
        Spatial shape of one frame.
    desc : dict | None
        Scaling stats: ``{'mean', 'std'}``.
    bool_mask : ndarray[bool] | None
        Valid-feature mask after flattening.
    kernel : str
        Kernel name used.
    kwargs : dict
        Extra parameters forwarded to ``KernelPCA``.
    """

    def __init__(self, comp, kernel="rbf", **kwargs):
        # comp is the number of components
        self.comp = comp  # TODO: default value passing
        self.components = None  # Transformed (projected) data
        self.pca = None  # Will hold the KernelPCA instance
        self.shape = None  # Shape of the original spatial grid
        self.desc = None  # Dictionary containing metadata (e.g., mean, std)
        self.bool_mask = None  # Mask for valid features
        self.kernel = kernel  # Kernel type for KernelPCA
        self.kwargs = kwargs  # Additional parameters for KernelPCA

        print("Using Kernel PCA")

    # Create setter method for class variables
    def set_from_simulation(self, sim):
        """Copy metadata from ``sim`` (see PCA variant)."""
        self.time_dim = sim.time_dim
        self.shape = sim.shape
        self.len = sim.len
        self.desc = sim.desc
        self.simulation = sim.simulation

    def get_num_components(self):
        """Return ``self.comp``."""
        return self.comp

    def decompose(self, simulation, length):
        """Run KernelPCA.

        Parameters
        ----------
        simulation : ndarray
            ``(time, *spatial_dims)``.
        length : int
            Number of time steps.

        Returns
        -------
        components : ndarray
        pca : KernelPCA
        bool_mask : ndarray[bool]
        """
        # Reshape the simulation data: assume simulation is
        # originally (time, height, width)
        array = simulation.reshape(length, -1)
        # Create a boolean mask of valid (finite) features
        self.bool_mask = np.asarray(np.isfinite(array[0, :]), dtype=bool)
        array_masked = array[:, self.bool_mask]
        # Save the original spatial shape (for later reconstruction)
        self.shape = simulation.shape[1:]
        # Instantiate KernelPCA with inverse transform enabled
        kpca = KernelPCA(
            n_components=self.comp,
            kernel=self.kernel,
            fit_inverse_transform=True,
            **self.kwargs,
        )
        # Fit and transform the masked data
        self.components = kpca.fit_transform(array_masked)
        self.pca = kpca
        return self.components, self.pca, self.bool_mask

    @staticmethod
    def reconstruct_predictions(predictions, n, info, begin=0):
        """Rebuild fields from predicted PCA scores.

        Parameters
        ----------
        predictions : pandas.DataFrame
            Rows = time, columns = components.
        n : int
            Number of components to use.
        info : dict
            Keys: ``'mask'``, ``'shape'``, ``'pca'``, ``'desc'``.
        begin : int, default 0
            First row to reconstruct.

        Returns
        -------
        int_mask : ndarray[int]
            ``info['mask']`` reshaped.
        rec : ndarray
            Reconstructed array, rescaled.
        """
        rec = []
        int_mask = info["mask"].astype(np.int32).reshape(info["shape"])

        # Reconstruct each year/time interval from components
        for t in range(begin, len(predictions)):
            # Create an array for the t-th prediction;
            # pad with zeros for any missing components
            arr = np.array(
                list(predictions.iloc[t, :n]) + [0] * (info["pca"].n_components - n)
            )
            map_ = np.zeros(info["shape"], dtype=np.float32)

            # Reshape arr to 2D (sample, n_components) for inverse_transform,
            # flatten inverse transform output, assign output to orginal ocean grid
            map_[int_mask == 1] = (
                info["pca"].inverse_transform(arr.reshape(1, -1)).flatten()
            )

            map_[int_mask == 0] = np.nan
            rec.append(map_)
        # Scale the reconstruction back using provided descriptors
        return int_mask, np.array(rec) * 2 * info["desc"]["std"] + info["desc"]["mean"]

    def reconstruct_components(self, n):
        """Reconstruct array with first ``n`` kernel PCs.

        Parameters
        ----------
        n : int
            Number of components retained.

        Returns
        -------
        ndarray
            ``(time, *self.shape)`` reconstruction.
        """
        rec = []
        # Convert the boolean mask to an integer mask and reshape to match original grid
        self.int_mask = self.bool_mask.astype(np.int32).reshape(self.shape)

        # Reconstruct each year/time interval from components
        for t in range(len(self.components)):
            arr = np.array(
                list(self.components[t, :n]) + [0] * (self.pca.n_components - n)
            )
            map_ = np.zeros(self.shape, dtype=np.float32)

            # Reshape arr to 2D (sample, n_components) for inverse_transform,
            # flatten inverse transform output, assign output to orginal ocean grid
            map_[self.int_mask == 1] = self.pca.inverse_transform(
                arr.reshape(1, -1)
            ).flatten()
            map_[self.int_mask == 0] = np.nan
            rec.append(map_)
        return np.array(rec)

    def get_component(self, n):
        """
        Get an approximate kernel principal component map for component n.

        For non-linear kernels the mapping is implicit, so this is only a proxy.

        Parameters
        ----------
        n : int
            Component index.

        Returns
        -------
        ndarray
            ``self.shape``: A 2D map corresponding to the n-th component.
        """
        # Create a flat map, fill with NaNs for invalid entries
        map_ = np.zeros(np.prod(self.shape), dtype=float)
        map_[~self.bool_mask] = np.nan
        # For linear kernels one might extract an interpretable component;
        # here we use the dual coefficients as a proxy (if available)
        if hasattr(self.pca, "alphas_") and n < self.pca.alphas_.shape[1]:
            map_[self.bool_mask] = self.pca.alphas_[:, n]
        else:
            # Otherwise, default to zeros if the component is not available
            map_[self.bool_mask] = 0
        map_ = map_.reshape(self.shape)
        # Scale the component using the stored descriptor parameters
        map_ = 2 * map_ * self.desc["std"] + self.desc["mean"]
        return map_

    def error(self, n):
        """Alias of :meth:`rmse`."""
        return self.rmse(n)

    def rmse(self, n):
        """RMSE using ``n`` kernel PCs.

        Parameters
        ----------
        n : int
            Components used in reconstruction.

        Returns
        -------
        reconstruction : ndarray
        rmse_values : ndarray
            Per-time RMSE.
        rmse_map : ndarray
            Per-point RMSE.
        """
        reconstruction = self.reconstruct_components(n)
        rmse_values = self.rmse_values(reconstruction) * 2 * self.desc["std"]
        rmse_map = self.rmse_map(reconstruction) * 2 * self.desc["std"]

        return reconstruction, rmse_values, rmse_map

    def rmse_values(self, reconstruction):
        """RMSE per time sample.

        Parameters
        ----------
        reconstruction : ndarray
            Output of :meth:`reconstruct_components`.

        Returns
        -------
        ndarray
            RMSE for each time index.
        """
        truth = self.simulation  # Assumes self.simulation is set externally
        rec = reconstruction
        if len(np.shape(truth)) == 3:
            valid_count = np.count_nonzero(~np.isnan(truth[0]))
            rmse_values = np.sqrt(
                np.nansum((truth - rec) ** 2, axis=(1, 2)) / valid_count
            )
        else:
            valid_count = np.count_nonzero(~np.isnan(self.simulation[0]), axis=(1, 2))
            rmse_values = np.nansum((truth - rec) ** 2, axis=(2, 3))
            for i in range(len(valid_count)):
                rmse_values[:, i] = rmse_values[:, i] / valid_count[i]
            rmse_values = np.sqrt(rmse_values)
        return rmse_values

    def rmse_map(self, reconstruction):
        """RMSE per spatial location.

        Parameters
        ----------
        reconstruction : ndarray
            Output of :meth:`reconstruct_components`.

        Returns
        -------
        ndarray
            ``self.shape`` RMSE map.
        """
        t = (
            self.len
        )  # Assumes self.len is defined elsewhere (e.g., number of time steps)
        return np.sqrt(np.sum((self.simulation[:] - reconstruction) ** 2, axis=0) / t)


# Creates a dictionary of Dict[classname -> class] key, value pairs
dimensionality_reduction_techniques = {
    "PCA": DimensionalityReductionPCA,
    "KernelPCA": DimensionalityReductionKernelPCA,
    "cvae":DimensionalityReductionCVAE
}
