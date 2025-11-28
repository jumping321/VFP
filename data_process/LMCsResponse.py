import numpy as np
from scipy.signal import butter, lfilter
from typing import List, Dict

class LMCs: 
    def __init__(self, subtype=None):
        self.NEURON_GRID = (41, 41)
        self.TIME_STEP = 1
        self.L1_DECAY_TAU = 300
        self.L2_DECAY_TAU = 300
        self.L3_DECAY_TAU = 200
        self.LOW_PASS_CUTOFF = 10
        self.SAMPLING_RATE = 1000
        self.subtype = subtype
        self.b_lp, self.a_lp = butter(4, self.LOW_PASS_CUTOFF, btype='low', fs=self.SAMPLING_RATE)

    def load_weights(self, neuron_id, side='100_8'):
        weights = {}
        all_vals = []
        for layer in ['L1', 'L2', 'L3']:
            path = f'./results/matrices/neuron_matrices_centered/{layer}_{side}_{neuron_id}.npz'
            try:
                with np.load(path) as d:
                    exc = d.get('exc', np.zeros(self.NEURON_GRID))
                    inh = d.get('inh', np.zeros(self.NEURON_GRID))
                    all_vals.append(np.max(np.abs(exc)))
                    all_vals.append(np.max(np.abs(inh)))
            except:
                continue
        global_max = np.max(all_vals) if all_vals else 1.0
        for layer in ['L1', 'L2', 'L3']:
            path = f'./results/matrices/neuron_matrices_centered/{layer}_{side}_{neuron_id}.npz'
            try:
                with np.load(path) as d:
                    exc = d.get('exc', np.zeros(self.NEURON_GRID)).astype(np.float32) / (global_max + 1e-8)
                    inh = d.get('inh', np.zeros(self.NEURON_GRID)).astype(np.float32) / (global_max + 1e-8)
                    weights[layer] = {'exc': exc, 'inh': inh}
            except:
                weights[layer] = {'exc': np.zeros(self.NEURON_GRID), 'inh': np.zeros(self.NEURON_GRID)}
        return weights

    def _silu(self, x):
        return x * (1 / (1 + np.exp(-x)))

    class NeuronLayer:
        def __init__(self, layer_type):
            self.response = np.zeros((41, 41), dtype=np.float32)
            self.layer_type = layer_type
            self.static_component = np.zeros((41, 41), dtype=np.float32) if layer_type in ['L1', 'L3'] else None
            self.params = {
                'L1_B': {'a': -0.917, 'b': 1.992},
                'L1_D': {'a': -2.326, 'b': -5.377},
                'L2_B': {'a': -0.814, 'b': 1.950},
                'L2_D': {'a': -2.044, 'b': -3.623}
            }

        def _sigmoid(self, x, a, b):
            return a * x / (1 + np.abs(b * x))

        def _contrast(self, current, last):
            contrast = np.zeros_like(current)
            mask = last != 0
            contrast[mask] = (current[mask] - last[mask]) / last[mask]
            contrast[(last == 0) & (current > 0)] = 1e4
            return contrast

        def update(self, current, last, dt, tau):
            contrast = self._contrast(current, last)
            direction = np.where(current > last, 'B', 'D')
            delta = np.zeros_like(contrast)
            if self.layer_type == 'L1':
                self.static_component = 0.35 * np.exp(-2.36 * current) + 0.08
                for d in ['B', 'D']:
                    mask = (direction == d)
                    p = self.params[f'L1_{d}']
                    delta[mask] = self._sigmoid(contrast[mask], p['a'], p['b'])
                self.response = self.response * np.exp(-dt / tau) + delta
                return self.response + self.static_component
            elif self.layer_type == 'L2':
                for d in ['B', 'D']:
                    mask = (direction == d)
                    p = self.params[f'L2_{d}']
                    delta[mask] = self._sigmoid(contrast[mask], p['a'], p['b'])
                self.response = self.response * np.exp(-dt / tau) + delta
                return self.response
            elif self.layer_type == 'L3':
                self.static_component = 0.62 * np.exp(-2.90 * current) + 0.08
                return self.static_component

    def calculate_response(self, stim: np.ndarray, weights: Dict):
        n_steps = stim.shape[2]
        layers = {name: self.NeuronLayer(name) for name in ['L1', 'L2', 'L3']}
        last_stim = stim[:, :, 0].astype(np.float32)
        full_responses = np.zeros(n_steps, dtype=np.float32)

        for t in range(n_steps):
            current_stim = stim[:, :, t].astype(np.float32)
            layer_data = {
                name: layer.update(current_stim, last_stim, self.TIME_STEP, getattr(self, f'{name}_DECAY_TAU'))
                for name, layer in layers.items()
            }
            raw_response = sum(
                np.sum(layer_data[name] * weights[name]['exc']) + np.sum(layer_data[name] * weights[name]['inh'])
                for name in ['L1', 'L2', 'L3']
            )
            full_responses[t] = raw_response
            last_stim = current_stim

        lp_responses = lfilter(self.b_lp, self.a_lp, full_responses)
        final_responses = self._silu(lp_responses)
        return final_responses
    
    def calculate_layer_responses(self, stim: np.ndarray):
        """
        Calculate the average response of each layer (L1, L2, L3) over the neuron grid
        and return a dictionary containing responses for plotting.
        """
        n_steps = stim.shape[2]
        layers = {name: self.NeuronLayer(name) for name in ['L1', 'L2', 'L3']}
        last_stim = stim[:, :, 0].astype(np.float32)
        layer_responses = {name: np.zeros(n_steps, dtype=np.float32) for name in layers}

        for t in range(n_steps):
            current_stim = stim[:, :, t].astype(np.float32)
            layer_data = {
                name: layer.update(current_stim, last_stim, self.TIME_STEP, getattr(self, f'{name}_DECAY_TAU'))
                for name, layer in layers.items()
            }

            # Take mean over neuron grid instead of applying weights
            for name in layers:
                layer_responses[name][t] = np.mean(layer_data[name])

            last_stim = current_stim

        # Apply low-pass filter and activation for each layer
        for name in layers:
            lp_resp = lfilter(self.b_lp, self.a_lp, layer_responses[name])
            layer_responses[name] = self._silu(lp_resp)

        return layer_responses
