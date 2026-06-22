import numpy as np
import os
from pathlib import Path
import h5py
from copy import deepcopy
import torch
from sklearn.preprocessing import StandardScaler
import pickle


def load_pickle_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle_data(data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def save_data(data, save_path, id='', save=True):
    if not save:
        return
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    if isinstance(data, list):
        for i in range(len(data)):
            file_name = f"{save_path}/{id}_chunk{i}_motion.npy"
            np.save(file_name, data[i])
    else:
        file_name = f"{save_path}/{id}.npy"
        np.save(file_name, data)
    

def slice_data(data, FPS, window_size=5, stride=0.5, mode='overlap', save=False, save_path=None, id=None):
    """
    Slice data into windows of size window_size with a stride of stride. 
    Inputs:
        data: (T, D) Time x Dimension
        FPS: FPS of data
        window_size: Window size in seconds
        stride: Stride in seconds
        mode: Mode for the last window if the last window is not full
            'none': No action
            'pad': Zero pad 
            'discard': Discard the last window 
            'overlap': Overlap the last window with the previous window
        save: Whether to save the sliced data
        save_path: Path to save the sliced data
    Output:
        sliced_data: (N, window_size, D) Number of windows x Window size x Dimension
    """
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    total_len = data.shape[0]
    ws_len = int(window_size * FPS)
    s_len = int(stride * FPS)
    num_chunks = (total_len - ws_len) // s_len + 1

    sliced_data = []
    for i in range(num_chunks):
        s_idx = i * s_len
        e_idx = s_idx + ws_len
        sliced_data.append(data[s_idx:e_idx])

    if num_chunks * s_len + ws_len == total_len or mode == 'discard':
        save_data(sliced_data, save_path, id, save)
        return sliced_data
    
    last_chunk = np.zeros((ws_len, data.shape[1]))
    if mode == 'pad':    
        last_chunk[:total_len - num_chunks * s_len] = data[num_chunks * s_len:]
    elif mode == 'none':
        last_chunk = data[num_chunks * s_len:]
    elif mode == 'overlap':
        last_chunk = data[total_len - ws_len:]
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    sliced_data.append(last_chunk)

    save_data(sliced_data, save_path, id, save)
    return sliced_data

def chop_data(data, s_idx, e_idx):
    return data[s_idx:e_idx]

def slice_datas(datas, window_size=5, stride=0.5, mode='overlap', save1=False, save2=False, save_path=None, id=None):
    data1 = datas[0]['data']
    data2 = datas[1]['data']
    FPS1 = datas[0]['FPS']
    FPS2 = datas[1]['FPS']

    if np.abs(data1.shape[0] / FPS1 - data2.shape[0] / FPS2) > 0.1:
        print(f"Warning: {id} has different length")
        return

    if len(data1.shape) == 1:
        data1 = data1.reshape(-1, 1)
    if len(data2.shape) == 1:
        data2 = data2.reshape(-1, 1)

    total_time = min(data1.shape[0] / FPS1, data2.shape[0] / FPS2)
    total_len = data1.shape[0]
    ws_len_ref = int(window_size * FPS1)
    s_len_ref = int(stride * FPS1)
    ws_len_second = int(window_size * FPS2)
    s_len_second = int(stride * FPS2)
    num_chunks = (total_len - ws_len_ref) // s_len_ref + 1

    sliced_data1 = []
    sliced_data2 = []
    for i in range(num_chunks):
        # e_time = s_time + window_size
        sliced_data1.append(chop_data(data1, i * s_len_ref, i * s_len_ref + ws_len_ref))
        sliced_data2.append(chop_data(data2, i * s_len_second, i * s_len_second + ws_len_second))

    if num_chunks * s_len_ref + ws_len_ref == total_len or mode == 'discard':
        # save_data(sliced_data1, save_path, id, save1)
        save_data(sliced_data2, save_path, id, save2)
        return sliced_data1, sliced_data2
    
    last_chunk1 = np.zeros((ws_len_ref, data1.shape[1]))
    last_chunk2 = np.zeros((ws_len_second, data2.shape[1]))
    s_time = num_chunks * stride
    if mode == 'pad':
        last_chunk1[:int((total_time - s_time) * FPS1)] = data1[int(s_time * FPS1):]
        last_chunk2[:int((total_time - s_time) * FPS2)] = data2[int(s_time * FPS2):]
    elif mode == 'none':
        last_chunk1 = data1[int(s_time * FPS1):]
        last_chunk2 = data2[int(s_time * FPS2):]
    elif mode == 'overlap':
        last_chunk1 = data1[int(total_time * FPS1) - ws_len_ref:int(total_time * FPS1)]
        last_chunk2 = data2[int(total_time * FPS2) - ws_len_second:int(total_time * FPS2)]
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    sliced_data1.append(last_chunk1)
    sliced_data2.append(last_chunk2)

    # save_data(sliced_data1, save_path, id, save1)
    save_data(sliced_data2, save_path, id, save2)
    return sliced_data1, sliced_data2

def _handle_zeros_in_scale(scale, copy=True, constant_mask=None):
    # if we are fitting on 1D arrays, scale might be a scalar
    if constant_mask is None:
        # Detect near constant values to avoid dividing by a very small
        # value that could lead to surprising results and numerical
        # stability issues.
        constant_mask = scale < 10 * torch.finfo(scale.dtype).eps

    if copy:
        # New array to avoid side-effects
        scale = scale.clone()
    scale[constant_mask] = 1.0
    return scale


class MinMaxScaler:
    _parameter_constraints: dict = {
        "feature_range": [tuple],
        "copy": ["boolean"],
        "clip": ["boolean"],
    }

    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip

    def _reset(self):
        if hasattr(self, "scale_"):
            del self.scale_
            del self.min_
            del self.n_samples_seen_
            del self.data_min_
            del self.data_max_
            del self.data_range_

    def fit(self, X):
        self._reset()
        return self.partial_fit(X)
    
    def partial_fit(self, X):
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                "Minimum of desired feature range must be smaller than maximum. Got %s."
                % str(feature_range)
            )

        data_min = torch.min(X, axis=0)[0]
        data_max = torch.max(X, axis=0)[0]

        self.n_samples_seen_ = X.shape[0]
        data_range = data_max - data_min
        self.scale_ = (feature_range[1] - feature_range[0]) / _handle_zeros_in_scale(
            data_range, copy=True
        )
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X):
        X *= self.scale_.to(X.device)
        X += self.min_.to(X.device)
        if self.clip:
            torch.clip(X, self.feature_range[0], self.feature_range[1], out=X)
        return X

    def inverse_transform(self, X):
        X -= self.min_[-X.shape[1] :].to(X.device)
        X /= self.scale_[-X.shape[1] :].to(X.device)
        return X


# class Normalizer:
#     def __init__(self, data):
#         flat = data.reshape(-1, data.shape[-1])
#         self.scaler = MinMaxScaler((-1, 1), clip=True)
#         self.scaler.fit(flat)

#     def normalize(self, x):
#         if x.ndim == 2:
#             return self.scaler.transform(x)
#         elif x.ndim == 3:
#             batch, seq, ch = x.shape
#             x = x.reshape(-1, ch)
#             return self.scaler.transform(x).reshape((batch, seq, ch))
#         else:
#             raise ValueError(f"Invalid shape: {x.shape}")
        

#     def unnormalize(self, x):
#         if x.ndim == 2:
#             x = torch.clip(x, -1, 1)  # clip to force compatibility
#             return self.scaler.inverse_transform(x)
#         elif x.ndim == 3:
#             batch, seq, ch = x.shape
#             x = x.reshape(-1, ch)
#             x = torch.clip(x, -1, 1)  # clip to force compatibility
#             return self.scaler.inverse_transform(x).reshape((batch, seq, ch))
#         else:
#             raise ValueError(f"Invalid shape: {x.shape}")

class Normalizer:
    def __init__(self, data=None, path=None):
        if data is not None:
            flat = data.reshape(-1, data.shape[-1])
            self.scaler = StandardScaler()
            self.scaler.fit(flat)
            self.mean = torch.from_numpy(self.scaler.mean_) #.mean()
            self.std = torch.from_numpy(self.scaler.scale_) #.mean()
        elif path is not None:
            self.load_normalizer(path)
        else:
            raise ValueError("Either data or path must be provided")

    def load_normalizer(self, path):
        data = np.load(path, allow_pickle=True)[()]
        self.scaler = StandardScaler()
        self.scaler.mean_ = data['mean']
        self.scaler.scale_ = data['std']
        self.mean = torch.from_numpy(self.scaler.mean_) #.mean()
        self.std = torch.from_numpy(self.scaler.scale_) #.mean()

    def save_normalizer(self, path):
        data = {
            'mean': self.scaler.mean_,
            'std': self.scaler.scale_
        }
        np.save(path, data)

    def normalize(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.ndim == 2:
            normalized = (x - self.mean.type_as(x)) / self.std.type_as(x)
            return normalized
            # return torch.from_numpy(self.scaler.transform(x)).type_as(x)
        elif x.ndim == 3:
            batch, seq, ch = x.shape
            x = x.reshape(-1, ch)
            normalized = (x - self.mean.type_as(x)) / self.std.type_as(x)
            return normalized.reshape((batch, seq, ch))
            # return torch.from_numpy(self.scaler.transform(x)).type_as(x).reshape((batch, seq, ch))
        else:
            raise ValueError(f"Invalid shape: {x.shape}")
        
    def unnormalize(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.ndim == 2:
            return x * self.std.type_as(x) + self.mean.type_as(x)
        elif x.ndim == 3:
            batch, seq, ch = x.shape
            x = x.reshape(-1, ch)
            x = x * self.std.type_as(x) + self.mean.type_as(x)
            return x.reshape((batch, seq, ch))
        else:
            raise ValueError(f"Invalid shape: {x.shape}")

