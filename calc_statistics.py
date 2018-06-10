import numpy as np
import torch

class rmse(object):
    def __init__(self):
        self.num_observations = 0
        self.sum_square_observations = 0
        self.sum_abs_observations = 0


    def add(self,observations):
        self.sum_square_observations += (observations ** 2).sum()
        self.num_observations += observations.size(0)
        self.sum_abs_observations += torch.abs(observations).sum()

    def get_rmse(self):
        return np.sqrt(np.true_divide(self.sum_square_observations, self.num_observations))

    def get_normalized_rmse(self):
        unNorm_rmse = self.get_rmse()
        to_norm_val = np.true_divide(self.sum_abs_observations, self.num_observations)
        return np.true_divide(unNorm_rmse,to_norm_val )

    def get_mae(self):
        return np.true_divide(self.sum_abs_observations, self.num_observations)

    def reset(self):
        self.__init__()

