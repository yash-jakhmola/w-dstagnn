import numpy as np
import pandas as pd

def load_weighted_adjacency_matrix(file_path, num_v):
    df = pd.read_csv(file_path, header=None)
    df = df.to_numpy()
    df = np.float64(df > 0)
    return df

def load_PA(file_path):
    df = pd.read_csv(file_path, header=None)
    df = df.to_numpy()
    df = np.float64(df>0)
    return df
