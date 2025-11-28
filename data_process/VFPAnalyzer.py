import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import warnings

class VFPAnalyzer:
    def __init__(self, matrix_npz_path, neuron_ids_path, neuron_types_path):
        """
        Initialize analyzer (supports only new .npz format)
        :param matrix_npz_path: Path to sparse matrix .npz file (must contain W_data/W_indices/W_indptr/W_shape)
        :param neuron_ids_path: Path to neuron ID list .npy file
        :param neuron_types_path: Path to neuron type CSV file
        """
        # Load sparse matrix (enforce new format key names)
        try:
            loaded = np.load(matrix_npz_path)
            self.W = csr_matrix(
                (loaded['data'], loaded['indices'], loaded['indptr']),
                shape=loaded['shape']
            )
        except KeyError as e:
            available_keys = loaded.files if 'loaded' in locals() else []
            raise ValueError(
                f"NPZ file format error! Required keys: W_data, W_indices, W_indptr, W_shape\n"
                f"Found keys: {available_keys}"
            ) from e

        # Load neuron ID mapping
        self.neuron_ids = np.load(neuron_ids_path, allow_pickle=True)
        self.id_to_idx = {nid: idx for idx, nid in enumerate(self.neuron_ids)}
        self.idx_to_id = {idx: nid for idx, nid in enumerate(self.neuron_ids)}
        
        # Load neuron type data
        self.neuron_types_df = pd.read_csv(neuron_types_path)
        self._validate_neuron_types()

    def _validate_neuron_types(self):
        """Validate neuron type data integrity"""
        required_columns = {'root_id', 'type', 'side'}
        if not required_columns.issubset(self.neuron_types_df.columns):
            missing = required_columns - set(self.neuron_types_df.columns)
            raise ValueError(f"Neuron type file missing required columns: {missing}")

    def get_neuron_ids_by_type(self, neuron_type, side='right'):
        """
        Get neuron IDs of specified type
        :param neuron_type: Neuron type (e.g., L1/L2/L3)
        :param side: Brain side (left/right)
        :return: List of neuron IDs
        """
        df = self.neuron_types_df
        filtered = df[(df['type'] == neuron_type) & (df['side'] == side)]
        return filtered['root_id'].tolist()

    def set_blocked_types(self, blocked_types, side='right'):
        """
        Set neuron types to block (e.g., ['Tm3', 'T5b']),
        these nodes will not be allowed in path traversal.
        """
        df = self.neuron_types_df
        # Select all root_id belonging to blocked types on the specified side
        blocked = df[(df['type'].isin(blocked_types)) & (df['side'] == side)]['root_id'].tolist()
        # Convert to index set
        blocked_idx = {self.id_to_idx[nid] for nid in blocked if nid in self.id_to_idx}
        self.blocked_idx = blocked_idx

    def _single_source_search(self, args):
        source_id, max_depth, min_weight = args
        
        pos_weights = np.zeros(len(self.neuron_ids), dtype=np.float32)
        neg_weights = np.zeros(len(self.neuron_ids), dtype=np.float32)
        if source_id not in self.id_to_idx:
            warnings.warn(f"Source ID {source_id} not found in matrix")
            return pos_weights, neg_weights
        source_idx = self.id_to_idx[source_id]

        # If the source itself is blocked, return empty immediately
        if hasattr(self, 'blocked_idx') and source_idx in self.blocked_idx:
            return pos_weights, neg_weights

        from collections import deque
        queue = deque()
        queue.append((source_idx, 1.0, 0))
        visited_edges = set()

        while queue:
            current_idx, current_weight, depth = queue.popleft()

            if depth > max_depth or abs(current_weight) < min_weight:
                continue

            # If current_idx is blocked, do not record and do not expand further
            if hasattr(self, 'blocked_idx') and current_idx in self.blocked_idx:
                continue

            # Record weight (only if not blocked)
            if current_weight > 0:
                pos_weights[current_idx] += current_weight
            else:
                neg_weights[current_idx] += current_weight

            # Traverse outgoing edges
            for j in range(self.W.indptr[current_idx], self.W.indptr[current_idx + 1]):
                post_idx = self.W.indices[j]
                conn_weight = self.W.data[j]
                edge_key = (current_idx, post_idx)
                if edge_key in visited_edges:
                    continue
                visited_edges.add(edge_key)

                # If the target node is blocked, do not enqueue
                if hasattr(self, 'blocked_idx') and post_idx in self.blocked_idx:
                    continue

                new_weight = current_weight * conn_weight
                if abs(new_weight) >= min_weight:
                    queue.append((post_idx, new_weight, depth + 1))

        return pos_weights, neg_weights

    def compute_and_save_weights(self, neuron_types, side='right', blocked_types=None,
                                 max_depth=50, min_weight=1e-6, num_processes=None, chunk_size=10):
        """
        Add blocked_types parameter to block specified neuron types in path traversal.
        """
        if blocked_types is not None:
            self.set_blocked_types(blocked_types, side)

        # Same as original logic, but blocked_types passed through instance variable
        os.makedirs("./results", exist_ok=True)
        num_processes = num_processes or cpu_count()

        for neuron_type in neuron_types:
            print(f"\n{'='*50}\nProcessing {neuron_type} neurons (with block={blocked_types})\n{'='*50}")

            source_ids = self.get_neuron_ids_by_type(neuron_type, side)
            if not source_ids:
                print(f"Warning: No {neuron_type} neurons found")
                continue

            n_sources = len(source_ids)
            n_targets = len(self.neuron_ids)
            pos_matrix = np.zeros((n_sources, n_targets), dtype=np.float32)
            neg_matrix = np.zeros((n_sources, n_targets), dtype=np.float32)

            tasks = [(src_id, max_depth, min_weight) for src_id in source_ids]
            with Pool(processes=num_processes) as pool:
                results = list(tqdm(
                    pool.imap(self._single_source_search, tasks, chunksize=chunk_size),
                    total=n_sources,
                    desc=f"Processing {neuron_type}"
                ))
            for i, (pos_res, neg_res) in enumerate(results):
                pos_matrix[i] = pos_res
                neg_matrix[i] = neg_res

            # Save results, include blocked_types info in filename
            blocked_tag = "_noblock" if blocked_types is None else "_block" + "_".join(blocked_types)
            base_path = f"./results/{neuron_type.lower()}{blocked_tag}"

            np.savez(
                f"{base_path}_excitatory.npz",
                source_ids=source_ids,
                target_ids=self.neuron_ids,
                weight_matrix=pos_matrix,
                metadata={
                    'neuron_type': neuron_type,
                    'side': side,
                    'max_depth': max_depth,
                    'min_weight': min_weight,
                    'blocked_types': blocked_types
                }
            )
            np.savez(
                f"{base_path}_inhibitory.npz",
                source_ids=source_ids,
                target_ids=self.neuron_ids,
                weight_matrix=neg_matrix,
                metadata={
                    'neuron_type': neuron_type,
                    'side': side,
                    'max_depth': max_depth,
                    'min_weight': min_weight,
                    'blocked_types': blocked_types
                }
            )
            print(f"Results saved to {base_path}_excitatory.npz and {base_path}_inhibitory.npz")
