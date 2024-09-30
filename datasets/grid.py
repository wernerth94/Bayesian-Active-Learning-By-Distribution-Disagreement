import numpy as np
import torch
from core.data import BaseDataset, DensityDataset, normalize, to_torch, to_one_hot
from sklearn.model_selection import train_test_split
import torchvision
from torch.utils.data import Dataset


class Grid(BaseDataset):

    def __init__(self, cache_folder:str, config:dict, pool_rng, encoded:bool,
                 data_file="grid.mock",):
        raise NotImplementedError()
        super().__init__(cache_folder, config, pool_rng, encoded, data_file)


    def _create_data(self, n_clsters=10, n_samples_per_clstr=50):
        x = torch.rand(size=(n_samples, dims)) * 8 - 4 # [-4, 4]
        y = torch.sin(x)
        noise = torch.randint(2, size=y.size()) - 0.5 # [-0.5, 0.5]
        noise += torch.normal(0, 0.2, size=noise.size())
        y += noise
        return x, y


    def _download_data(self, dataset='ThreeClust', train_ratio=0.8, test_ratio=0.20):
        raise NotImplementedError


    def _load_data(self, dataset='ThreeClust', train_ratio=0.8, test_ratio=0.20):
        assert train_ratio + test_ratio == 1, "The sum of train, val, and test should be equal to 1."

        x, y = self._create_data()

        ids = np.arange(x.shape[0], dtype=int)
        self.pool_rng.shuffle(ids)
        cut = int(len(ids) * test_ratio)
        train_ids = ids[cut:]
        test_ids = ids[:cut]

        x_train = x[train_ids]
        y_train = y[train_ids]
        x_test = x[test_ids]
        y_test = y[test_ids]

        x_train, x_test = normalize(x_train, x_test, mode="min_max")
        y_train, y_test = normalize(y_train.reshape(-1, 1), y_test.reshape(-1, 1), mode="min_max")
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


class DensityGrid(DensityDataset):

    def __init__(self, cache_folder:str, config:dict, pool_rng, encoded=False,
                 data_file="density_bi_modal_sin.mock",):
        super().__init__(cache_folder, config, pool_rng, data_file)


    def _create_data(self, n_clsters=4, n_samples_per_clstr=100):
        X = np.zeros((0, 2))
        for y in range(n_clsters):
            for x in range(n_clsters):
                sample = np.random.multivariate_normal((x * 2, y * 2), np.eye(2) * 0.1, size=n_samples_per_clstr)
                X = np.concatenate((X, sample), axis=0)
        return X


    def _download_data(self, dataset='ThreeClust', train_ratio=0.8, test_ratio=0.20):
        raise NotImplementedError


    def _load_data(self, dataset='', train_ratio=0.8, test_ratio=0.20):
        assert train_ratio + test_ratio == 1, "The sum of train, val, and test should be equal to 1."

        x = self._create_data()

        ids = np.arange(x.shape[0], dtype=int)
        self.pool_rng.shuffle(ids)
        cut = int(len(ids) * test_ratio)
        train_ids = ids[cut:]
        test_ids = ids[:cut]

        x_train = x[train_ids]
        x_test = x[test_ids]

        x_train, x_test = normalize(x_train, x_test, mode="min_max")
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        return to_torch(x_train, torch.float32, device=self.device), \
               to_torch(x_test, torch.float32, device=self.device)
