import numpy as np
import torch
from core.data import BaseDataset, normalize, to_torch, to_one_hot
from sklearn.model_selection import train_test_split
import torchvision
from torch.utils.data import Dataset


class SynthData(BaseDataset):

    def __init__(self, cache_folder:str, config:dict, pool_rng, encoded:bool,
                 data_file=None, dataset='ThreeClust'):
        assert not encoded, "This dataset does not support encodings"
        self.dataset = dataset
        super().__init__(cache_folder, config, pool_rng, encoded, data_file)



    def createToy_ThreeClust(self, n_perClust=150, cov=[[1, 0], [0, 1]] ):

        mean1 = [0, 0]
        cluster1 = self.pool_rng.multivariate_normal(mean1, cov, n_perClust)

        mean2 = [4, 3]
        cluster2 = self.pool_rng.multivariate_normal(mean2, cov, n_perClust)

        mean3 = [0, 6]
        cluster3 = self.pool_rng.multivariate_normal(mean3, cov, n_perClust)

        mean4 = [4, 3]
        cluster4 = self.pool_rng.multivariate_normal(mean4, cov, n_perClust)

        data_pos = np.concatenate((cluster1, cluster2), axis=0)
        data_neg = np.concatenate((cluster3, cluster4), axis=0)

        data_pos = np.c_[data_pos, np.ones(len(data_pos))]
        data_neg = np.c_[data_neg, np.zeros(len(data_neg))]

        return np.concatenate((data_pos, data_neg), axis=0)


    def createDivergingSin(self, n_samples=1000, divergence_factor=0.5, sin_freq=2, cov=0.3):

        x = np.linspace(0, 10, n_samples)
        sin_curve = np.sin(sin_freq*x)

        # Cluster above the curve
        cluster_above_x = x
        cluster_above_y = sin_curve + divergence_factor * x + self.pool_rng.normal(0, cov, n_samples)
        cluster_above = np.c_[cluster_above_x, cluster_above_y]

        # Cluster below the curve
        cluster_below_x = x
        cluster_below_y = sin_curve - divergence_factor * x + self.pool_rng.normal(0, cov, n_samples)
        cluster_below = np.c_[cluster_below_x, cluster_below_y]


        data_pos = np.c_[cluster_above, np.ones(len(cluster_above_y))]
        data_neg = np.c_[cluster_below, np.zeros(len(cluster_below_y))]

        return np.concatenate((data_pos, data_neg), axis=0)

    def _download_data(self, dataset='ThreeClust', train_ratio=0.8, test_ratio=0.20):
        raise NotImplementedError

    def _load_data(self, dataset='ThreeClust', train_ratio=0.8, test_ratio=0.20):
        assert train_ratio + test_ratio == 1, "The sum of train, val, and test should be equal to 1."

        if self.dataset == 'ThreeClust':
            data = self.createToy_ThreeClust()
        elif self.dataset == 'DivergingSin':
            data = self.createDivergingSin()
        else:
            raise NotImplementedError

        ids = np.arange(data.shape[0], dtype=int)
        self.pool_rng.shuffle(ids)
        cut = int(len(ids) * test_ratio)
        train_ids = ids[cut:]
        test_ids = ids[:cut]

        x_train = data[train_ids, :2]
        y_train = to_one_hot(data[train_ids, -1].astype(int))
        x_test = data[test_ids, :2]
        y_test = to_one_hot(data[test_ids, -1].astype(int))

        x_train, x_test = normalize(x_train, x_test, mode="min_max")
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        return to_torch(x_train, torch.float32, device=self.device), \
               to_torch(y_train, torch.float32, device=self.device), \
               to_torch(x_test, torch.float32, device=self.device), \
               to_torch(y_test, torch.float32, device=self.device),


    # Abstract methods from the Parent
    def get_pretext_transforms(self, config: dict) -> torchvision.transforms.Compose:
        raise NotImplementedError

    def get_pretext_validation_transforms(self, config: dict) -> torchvision.transforms.Compose:
        raise NotImplementedError

    def load_pretext_data(self) -> tuple[Dataset, Dataset]:
        raise NotImplementedError


class ThreeClust(SynthData):
    def __init__(self, cache_folder:str, config:dict, pool_rng, encoded:bool,
                 data_file=None, dataset='ThreeClust'):
        super().__init__(cache_folder, config, pool_rng, encoded,
                         data_file, dataset)


class DivergingSin(SynthData):
    def __init__(self, cache_folder:str, config:dict, pool_rng, encoded:bool,
                 data_file=None, dataset='DivergingSin'):
        super().__init__(cache_folder, config, pool_rng, encoded,
                         data_file, dataset)

