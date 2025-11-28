import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from joblib import Parallel, delayed
from tqdm import tqdm

class WeightMatrixExporter:
    def __init__(self, connections_path, synapses_path, 
                 target_pos_sum=1.0, target_neg_sum=-1.0, n_jobs=-1, results_dir="results"):
        self.connections_path = connections_path
        self.synapses_path = synapses_path
        self.target_pos_sum = target_pos_sum
        self.target_neg_sum = target_neg_sum
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.results_dir = results_dir
        self.W = None
        self.neuron_ids = []
        self.id_to_idx = {}

        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)

    def load_data(self):
        conn_df = pd.read_csv(self.connections_path, 
                            usecols=['pre_root_id', 'post_root_id', 'syn_count', 'nt_type'])
        synapses_df = pd.read_csv(self.synapses_path, 
                                usecols=['root_id', 'input synapses'])
        input_synapses = synapses_df.set_index('root_id')['input synapses'].to_dict()
        return conn_df, input_synapses

    def build_weight_matrix(self):
        conn_df, input_synapses = self.load_data()
        
        # Build ID map (ensure neuron_ids are sorted)
        all_ids = set(conn_df['pre_root_id']).union(set(conn_df['post_root_id']))
        self.neuron_ids = sorted(all_ids)  # ensure sorting
        self.id_to_idx = {n: i for i, n in enumerate(self.neuron_ids)}
        
        # Stage 1: parallel computation of raw weights
        print("Computing raw weights...")
        W_raw = self._build_raw_weights_parallel(conn_df, input_synapses)
        
        # Stage 2: memory-efficient parallel normalization
        print("Normalizing weights...")
        self.W = self._normalize_weights_sparse(W_raw)
        return self.W

    def _build_raw_weights_parallel(self, conn_df, input_synapses):
        # Chunked parallel processing
        def process_chunk(chunk):
            rows, cols, data = [], [], []
            for _, row in chunk.iterrows():
                pre_idx = self.id_to_idx[row['pre_root_id']]
                post_idx = self.id_to_idx[row['post_root_id']]
                sign = -1 if row['nt_type'] in {"GABA", "GLUT"} else 1
                post_input_syn = input_synapses.get(row['post_root_id'], 1)  # Default to avoid KeyError
                weight = np.tanh(row['syn_count'] * sign / post_input_syn) # Prevent divide-by-zero
                rows.append(pre_idx)
                cols.append(post_idx)
                data.append(weight)
            return (rows, cols, data)

        chunks = np.array_split(conn_df, self.n_jobs * 4)
        results = Parallel(n_jobs=self.n_jobs)(delayed(process_chunk)(chunk) for chunk in tqdm(chunks))
        
        # Merge results
        all_rows, all_cols, all_data = [], [], []
        for r, c, d in results:
            all_rows.extend(r)
            all_cols.extend(c)
            all_data.extend(d)
            
        return coo_matrix((all_data, (all_rows, all_cols)), 
                         shape=(len(self.neuron_ids), len(self.neuron_ids))).tocsr()

    def _normalize_weights_sparse(self, W_raw):
        num_neurons = W_raw.shape[0]
        
        # If sums are 0, skip normalization
        if self.target_pos_sum == 0 and self.target_neg_sum == 0:
            print("Skipping normalization (both target sums are 0)")
            return W_raw
            
        # Compute per-column positive/negative sums (sparse operations)
        print("Calculating sums...")
        pos_sums = np.zeros(num_neurons)
        neg_sums = np.zeros(num_neurons)
        
        # Block-wise sum processing
        chunk_size = 1000  # process 1000 columns per chunk
        for i in tqdm(range(0, num_neurons, chunk_size)):
            chunk = W_raw[:, i:i+chunk_size]
            chunk_arr = chunk.toarray()  # convert small block to dense
            if self.target_pos_sum != 0:  # compute only if needed
                pos_sums[i:i+chunk_size] = np.sum(np.where(chunk_arr > 0, chunk_arr, 0), axis=0)
            if self.target_neg_sum != 0:  # compute only if needed
                neg_sums[i:i+chunk_size] = np.sum(np.where(chunk_arr < 0, chunk_arr, 0), axis=0)
        
        # Precompute scaling factors
        pos_scale = np.ones(num_neurons)  # default no scaling
        neg_scale = np.ones(num_neurons)  # default no scaling
        
        if self.target_pos_sum != 0:
            pos_scale = np.where(pos_sums > 0, self.target_pos_sum / pos_sums, 1)
        if self.target_neg_sum != 0:
            neg_scale = np.where(neg_sums < 0, self.target_neg_sum / neg_sums, 1)
        
        print("Applying normalization...")
        # Apply normalization in blocks
        normalized_data = []
        normalized_rows = []
        normalized_cols = []
        
        for i in tqdm(range(0, num_neurons, chunk_size)):
            chunk = W_raw[:, i:i+chunk_size]
            chunk_coo = chunk.tocoo()
            
            # Apply scaling
            for row, col, val in zip(chunk_coo.row, chunk_coo.col, chunk_coo.data):
                actual_col = i + col
                if val > 0 and self.target_pos_sum != 0:  # only scale if needed
                    scaled_val = val * pos_scale[actual_col]
                elif val < 0 and self.target_neg_sum != 0:  # only scale if needed
                    scaled_val = val * neg_scale[actual_col]
                else:
                    scaled_val = val
                
                normalized_rows.append(row)
                normalized_cols.append(actual_col)
                normalized_data.append(scaled_val)
        
        # Construct normalized sparse matrix
        return coo_matrix((normalized_data, (normalized_rows, normalized_cols)),
                         shape=(num_neurons, num_neurons)).tocsr()

    def save_weight_matrix_npy(self):
        """Save weight matrix into three separate files: sparse_matrix.npz + neuron_ids.npy + text file"""
        if self.W is None:
            self.build_weight_matrix()
            
        # 1. Save sparse matrix (.npz)
        sparse_filename = os.path.join(self.results_dir, f"weight_matrix.npz")
        np.savez(sparse_filename, 
                data=self.W.data, 
                indices=self.W.indices, 
                indptr=self.W.indptr,
                shape=self.W.shape)
        
        # 2. Save neuron IDs (.npy)
        ids_filename = os.path.join(self.results_dir, f"neuron_ids.npy")
        np.save(ids_filename, self.neuron_ids)
        
        # 3. Save text file (.txt)
        txt_filename = os.path.join(self.results_dir, f"weight_matrix.txt")
        self._save_as_txt(txt_filename)
        
        print(f"\nSparse weight matrix saved to: {sparse_filename}")
        print(f"Neuron IDs saved to: {ids_filename}")
        print(f"Text format saved to: {txt_filename}")

    def _save_as_txt(self, filename, max_entries=None):
        """Internal method: save to text file"""
        W_coo = self.W.tocoo()
        total_entries = len(W_coo.data)
        
        if max_entries is not None and max_entries < total_entries:
            print(f"Only saving top {max_entries:,} strongest connections (of {total_entries:,})")
            idx = np.argsort(-np.abs(W_coo.data))[:max_entries]
            rows = W_coo.row[idx]
            cols = W_coo.col[idx]
            data = W_coo.data[idx]
        else:
            rows = W_coo.row
            cols = W_coo.col
            data = W_coo.data
        
        # Write into file
        with open(filename, 'w') as f:
            # Header information
            f.write(f"# Weight Matrix (shape: {self.W.shape[0]}x{self.W.shape[1]})\n")
            f.write(f"# Total neurons: {len(self.neuron_ids)}\n")
            f.write(f"# Non-zero entries: {len(data):,}\n")
            f.write("# Format: pre_neuron_id post_neuron_id weight_value\n\n")
            
            # Write data
            for r, c, v in zip(rows, cols, data):
                pre_id = self.neuron_ids[r]
                post_id = self.neuron_ids[c]
                f.write(f"{pre_id} {post_id} {v:.6g}\n")
