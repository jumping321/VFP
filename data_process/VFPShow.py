import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import issparse
from tqdm import tqdm
import os
from scipy.ndimage import shift

class VFPShow:
    def __init__(self, coord_path: str):
        """
        Initialize heatmap generator.

        Args:
            coord_path: Path to neuron coordinates CSV. Expected columns include 'root_id', 'x', 'y'.
        """
        # Load coordinate table and build quick lookup
        self.coord_df = pd.read_csv(coord_path)
        self._build_coord_mapping()

        # Load neuron type table for lookups used elsewhere
        self.neuron_types_df = pd.read_csv('./data/visual_neuron_types.txt')

        # Colormaps for excitatory (pos) and inhibitory (neg) displays
        self.exc_cmap = LinearSegmentedColormap.from_list('exc', ['#FFFFFF', '#FF0000'])
        self.inh_cmap = LinearSegmentedColormap.from_list('inh', ['#FFFFFF', '#0000FF'])

        # Fixed plotting grid parameters
        self.xlim = (-20, 20)
        self.ylim = (-20, 20)
        self.resolution = 1

    def hex_to_cartesian(self, p: float, q: float) -> tuple:
        """
        Convert hex coordinates (p,q) to Cartesian (x,y).
        The conversion follows a simple odd-row offset scheme.
        """
        x = 2 * p + 1 if q % 2 == 1 else 2 * p
        y = q / 2
        return x, y

    def get_neuron_ids_by_type(self, neuron_type, side='right'):
        """
        Return list of neuron IDs for a given neuron type and brain side.

        Args:
            neuron_type: type label (e.g., 'L1', 'L2', 'L3' or cell classes)
            side: 'left' or 'right'
        """
        df = self.neuron_types_df
        filtered = df[(df['type'] == neuron_type) & (df['side'] == side)]
        return filtered['root_id'].tolist()

    def _build_coord_mapping(self):
        """
        Build a mapping from root_id to Cartesian coordinates.
        Rows with missing or invalid coordinates are skipped.
        """
        self.coord_map = {}
        for _, row in self.coord_df.iterrows():
            try:
                x, y = self.hex_to_cartesian(float(row['x']), float(row['y']))
                self.coord_map[int(row['root_id'])] = (x, y)
            except (ValueError, KeyError):
                # Skip rows with invalid data
                continue

    def generate_all_matrices(self, side: str):
        """
        Process all layers (L1, L2, L3) for the given hemisphere side.

        Workflow:
        - Load per-layer excitatory/inhibitory source->target matrices from:
            ./results/matrices/{layer}_{side}_excitatory.npz
            ./results/matrices/{layer}_{side}_inhibitory.npz
        - Rasterize each source neuron's contribution into a 2D grid based on coordinates.
        - Save the rasterized matrices (with coefficient applied) to:
            ./results/matrices/neuron_matrices{side}/
        - Compute the combined centroid across layers and center each raster; save to:
            ./results/matrices/neuron_matrices_centered{side}/

        Note: Scaling (scale_pos/scale_neg) and saving scaled resultss are intentionally omitted.
        """
        # Compute grid dimensions
        x_bins = int((self.xlim[1] - self.xlim[0]) / self.resolution) + 1
        y_bins = int((self.ylim[1] - self.ylim[0]) / self.resolution) + 1

        print(f"\nProcessing all layers for {side} hemisphere...")

        # Create results directories (keep original folder names)
        os.makedirs(f'./results/matrices/neuron_matrices{side}', exist_ok=True)
        os.makedirs(f'./results/matrices/neuron_matrices_centered{side}', exist_ok=True)

        # Load layer data
        all_data = {}
        layers = ['L1', 'L2', 'L3']
        for layer in layers:
            pos_path = f'./results/matrices/{layer}_{side}_excitatory.npz'
            neg_path = f'./results/matrices/{layer}_{side}_inhibitory.npz'
            try:
                exc_data = np.load(pos_path, allow_pickle=True)
                inh_data = np.load(neg_path, allow_pickle=True)
                all_data[layer] = {
                    'exc': exc_data['weight_matrix'],
                    'inh': inh_data['weight_matrix'],
                    'source_ids': exc_data['source_ids'],
                    'target_ids': exc_data['target_ids']
                }
            except FileNotFoundError:
                # Warn and skip layers that are missing
                print(f"Warning: Missing data for {layer}_{side}")
                continue

        if not all_data:
            # No layer data found — nothing to do
            return

        # Target neuron list is taken from the first available layer
        target_ids = next(iter(all_data.values()))['target_ids']

        # Coefficients applied to raw excitatory/inhibitory values (kept at 1.0 by default)
        exc_coeff = 1.0
        inh_coeff = 1.0

        # Process each target neuron
        for neuron_idx in tqdm(range(len(target_ids)), desc='Processing neurons'):
            neuron_id = target_ids[neuron_idx]

            # Raster matrices for each layer (exc and inh)
            matrices = {}
            for layer in layers:
                if layer not in all_data:
                    continue

                matrices[layer] = {
                    'exc': np.zeros((y_bins, x_bins), dtype=np.float32),
                    'inh': np.zeros((y_bins, x_bins), dtype=np.float32)
                }

                src_ids = all_data[layer]['source_ids']
                exc_mat = all_data[layer]['exc']
                inh_mat = all_data[layer]['inh']

                # Rasterize each source neuron's contribution
                for src_idx in range(len(src_ids)):
                    src_id = src_ids[src_idx]
                    if src_id not in self.coord_map:
                        continue

                    x, y = self.coord_map[src_id]
                    x_idx = int((x - self.xlim[0]) / self.resolution)
                    y_idx = int((y - self.ylim[0]) / self.resolution)

                    if 0 <= x_idx < x_bins and 0 <= y_idx < y_bins:
                        matrices[layer]['exc'][y_idx, x_idx] = exc_mat[src_idx, neuron_idx] * exc_coeff
                        matrices[layer]['inh'][y_idx, x_idx] = inh_mat[src_idx, neuron_idx] * inh_coeff

            # Save raw raster matrices (with coefficients applied)
            for layer in matrices:
                np.savez_compressed(
                    f'./results/matrices/neuron_matrices{side}/{layer}_{side}_{neuron_id}.npz',
                    exc=matrices[layer]['exc'].astype(np.float16),
                    inh=matrices[layer]['inh'].astype(np.float16)
                )

            # Compute combined weights across layers for centroid calculation
            combined_weights = np.zeros((y_bins, x_bins), dtype=np.float32)
            for layer in matrices:
                combined_weights += matrices[layer]['exc'] + matrices[layer]['inh']

            if np.sum(np.abs(combined_weights)) == 0:
                # No signal for this neuron — skip centering and next steps
                continue

            # Compute centroid using absolute weights
            y_indices, x_indices = np.indices((y_bins, x_bins))
            total_weight = np.sum(np.abs(combined_weights))
            x_centroid = np.sum(np.abs(combined_weights) * x_indices) / total_weight
            y_centroid = np.sum(np.abs(combined_weights) * y_indices) / total_weight

            # Determine integer shift to move centroid to grid center
            center_y = y_bins // 2
            center_x = x_bins // 2
            shift_y = int(round(center_y - y_centroid))
            shift_x = int(round(center_x - x_centroid))

            # Helper to shift matrices with constant padding
            def shift_matrix(mat):
                return shift(mat, shift=(shift_y, shift_x), mode='constant', cval=0)

            # Save centered rasters
            for layer in matrices:
                exc_shifted = shift_matrix(matrices[layer]['exc'])
                inh_shifted = shift_matrix(matrices[layer]['inh'])

                np.savez_compressed(
                    f'./results/matrices/neuron_matrices_centered{side}/{layer}_{side}_{neuron_id}.npz',
                    exc=exc_shifted.astype(np.float16),
                    inh=inh_shifted.astype(np.float16)
                )

    def plot_single_neuron_all_layers(self, neuron_id: int, side: str, matrices: str):
        """
        Plot connectivity heatmaps for a single neuron across L1/L2/L3.

        Args:
            neuron_id: Target neuron ID.
            side: Hemisphere side string used in filenames.
            matrices: Subfolder in ./results/matrices/ to load from (e.g., 'neuron_matricesright' or 'neuron_matrices_centeredright').
        """
        # Create figure with 3 rows (layers) and 2 columns (exc/inh)
        fig, axes = plt.subplots(3, 2, figsize=(6, 8))

        # Title for the figure
        fig.suptitle(f'Neuron {neuron_id}', y=0.98, fontsize=20, fontfamily='Times New Roman')

        # Custom display colormaps
        exc_color = '#6FBF73'  # green-like for excitatory display
        inh_color = '#E57373'  # red-like for inhibitory display
        exc_cmap = LinearSegmentedColormap.from_list('exc_cmap', ['white', exc_color])
        inh_cmap = LinearSegmentedColormap.from_list('inh_cmap', ['white', inh_color])

        # Center coordinates for marker placement
        center_x = (self.xlim[0] + self.xlim[1]) / 2
        center_y = (self.ylim[0] + self.ylim[1]) / 2

        # Iterate over layers and plot excitatory (left) and inhibitory (right)
        for i, layer in enumerate(['L1', 'L2', 'L3']):
            # Excitatory axis
            ax_exc = axes[i, 0]
            try:
                data = np.load(f'./results/matrices/{matrices}/{layer}_{side}_{neuron_id}.npz')
                exc_matrix = data['exc']

                # Draw heatmap
                ax_exc.pcolormesh(
                    np.linspace(self.xlim[0], self.xlim[1], exc_matrix.shape[1] + 1),
                    np.linspace(self.ylim[0], self.ylim[1], exc_matrix.shape[0] + 1),
                    exc_matrix,
                    cmap=exc_cmap,
                    vmin=0,
                    shading='auto'
                )

                # Draw grid lines for clarity
                x_edges = np.linspace(self.xlim[0], self.xlim[1], exc_matrix.shape[1] + 1)
                y_edges = np.linspace(self.ylim[0], self.ylim[1], exc_matrix.shape[0] + 1)
                for x in x_edges:
                    ax_exc.axvline(x, color='gray', linewidth=1, alpha=0.6)
                for y in y_edges:
                    ax_exc.axhline(y, color='gray', linewidth=1, alpha=0.6)

                # Draw center marker
                ax_exc.scatter(center_x, center_y, s=100, c='gold', edgecolor='k', marker='*', zorder=10, alpha=0.5)

                # If this neuron has coordinates, plot them
                if neuron_id in self.coord_map:
                    x_pt, y_pt = self.coord_map[neuron_id]
                    ax_exc.scatter(x_pt, y_pt, s=100, c='gold', edgecolor='k', marker='*', zorder=11)

            except FileNotFoundError:
                # If file missing, draw an empty grid to preserve layout
                x_edges = np.linspace(self.xlim[0], self.xlim[1], 11)
                y_edges = np.linspace(self.ylim[0], self.ylim[1], 11)
                for x in x_edges:
                    ax_exc.axvline(x, color='gray', linewidth=1, alpha=0.6)
                for y in y_edges:
                    ax_exc.axhline(y, color='gray', linewidth=1, alpha=0.6)
                ax_exc.scatter(center_x, center_y, s=100, c='gold', edgecolor='k', marker='*', zorder=10, alpha=0.5)

            # Inhibitory axis
            ax_inh = axes[i, 1]
            try:
                data = np.load(f'./results/matrices/{matrices}/{layer}_{side}_{neuron_id}.npz')
                inh_matrix = np.abs(data['inh'])

                ax_inh.pcolormesh(
                    np.linspace(self.xlim[0], self.xlim[1], inh_matrix.shape[1] + 1),
                    np.linspace(self.ylim[0], self.ylim[1], inh_matrix.shape[0] + 1),
                    inh_matrix,
                    cmap=inh_cmap,
                    vmin=0,
                    shading='auto'
                )

                x_edges = np.linspace(self.xlim[0], self.xlim[1], inh_matrix.shape[1] + 1)
                y_edges = np.linspace(self.ylim[0], self.ylim[1], inh_matrix.shape[0] + 1)
                for x in x_edges:
                    ax_inh.axvline(x, color='gray', linewidth=1, alpha=0.6)
                for y in y_edges:
                    ax_inh.axhline(y, color='gray', linewidth=1, alpha=0.6)

                ax_inh.scatter(center_x, center_y, s=100, c='gold', edgecolor='k', marker='*', zorder=10, alpha=0.5)
                if neuron_id in self.coord_map:
                    x_pt, y_pt = self.coord_map[neuron_id]
                    ax_inh.scatter(x_pt, y_pt, s=100, c='gold', edgecolor='k', marker='*', zorder=11)

            except FileNotFoundError:
                x_edges = np.linspace(self.xlim[0], self.xlim[1], 11)
                y_edges = np.linspace(self.ylim[0], self.ylim[1], 11)
                for x in x_edges:
                    ax_inh.axvline(x, color='gray', linewidth=1, alpha=0.6)
                for y in y_edges:
                    ax_inh.axhline(y, color='gray', linewidth=1, alpha=0.6)
                ax_inh.scatter(center_x, center_y, s=100, c='gold', edgecolor='k', marker='*', zorder=10, alpha=0.5)

        # Final formatting for all subplots
        for ax in axes.flat:
            ax.set_xticks([])        # hide x ticks
            ax.set_yticks([])        # hide y ticks
            ax.set_aspect('equal')   # equal aspect ratio
            # set thicker border lines for clarity
            for spine in ax.spines.values():
                spine.set_linewidth(2)

        plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
        plt.show()
