import copy
import numpy as np
from collections import Counter
from scipy.spatial.distance import hamming

class KnnImputerWithHamming:
    def __init__(self, features):
        self.features = copy.deepcopy(features)
        self.distance_matrix = []
    
    def calculate_hamming_distance(self):
        for i in self.features.tolist():
            a_vector = []
            for j in self.features.tolist():
                a_vector.append(hamming(i, j))
            self.distance_matrix.append(a_vector)

    def impute_data(self, n=30, threshold=0.8):
        for idx in range(self.features.shape[0]):
            dist_mat_array = np.array(self.distance_matrix[idx])
            below_threshold = dist_mat_array[dist_mat_array < threshold]
            sorted_matrix = np.argsort(below_threshold)
            sorted_matrix = np.delete(sorted_matrix, np.where(sorted_matrix == idx))
            
            nan_indices = np.where(self.features[idx] != self.features[idx])[0]
            for nan_idx in nan_indices:
                not_non_values = self.features[sorted_matrix[:n]][:, nan_idx]
                not_non_idx = np.where(not_non_values == not_non_values)[0]
                not_non_values = not_non_values[not_non_idx]
                most_common = Counter(not_non_values.tolist()).most_common(1)
                self.features[idx, nan_idx] = most_common[0][0]