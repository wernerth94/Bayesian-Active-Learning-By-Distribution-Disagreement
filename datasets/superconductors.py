import os
from os.path import exists
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from core.data import GaussianNoise, VectorToTensor
from sklearn.datasets import load_svmlight_file
from core.data import BaseDataset, VectorDataset, normalize, postprocess_svm_data
import requests, zipfile
import pandas as pd
import numpy as np

class Superconductors(BaseDataset):

    def __init__(self, cache_folder:str, config:dict, pool_rng, encoded:bool,
                 data_file="superconductors_al.pt"):
        self.raw_zip_file = os.path.join(cache_folder, "superconductors_raw.zip")
        self.raw_unzipped_file = os.path.join(cache_folder, "superconductors_raw/train.csv")
        super().__init__(cache_folder, config, pool_rng, encoded, data_file)


    def _download_data(self, target_to_one_hot=True):
        train_url = "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"

        if not exists(self.raw_zip_file):
            print("File size: 8MB")
            r = requests.get(train_url)
            with open(self.raw_zip_file, 'wb') as f:
                f.write(r.content)
            print("Download successful")

        if not exists(self.raw_unzipped_file):
            z = zipfile.ZipFile(self.raw_zip_file)
            z.extract('train.csv', os.path.join(self.cache_folder, "superconductors_raw"))

        if exists(self.raw_unzipped_file):
            data = pd.read_csv(self.raw_unzipped_file, header=0, index_col=None)
            y = data["critical_temp"].values
            x = data.drop(["critical_temp"], axis=1).values
            del data
            ids = np.arange(len(x))
            self.pool_rng.shuffle(ids)
            cut = int(0.8 * len(x))
            train_ids, test_ids = ids[:cut], ids[cut:]
            self.x_train, self.y_train = x[train_ids], y[train_ids]
            self.x_test, self.y_test = x[test_ids], y[test_ids]
            self.x_train, self.x_test = normalize(self.x_train, self.x_test, mode="min_max")
            self.y_train, self.y_test = normalize(self.y_train.reshape(-1, 1), self.y_test.reshape(-1, 1), mode="min_max")
            self._convert_data_to_tensors()



    def load_pretext_data(self)->tuple[Dataset, Dataset]:
        raise NotImplementedError()

    def get_pretext_transforms(self, config:dict)->transforms.Compose:
        raise NotImplementedError()

    def get_pretext_validation_transforms(self, config:dict)->transforms.Compose:
        raise NotImplementedError()

    def get_meta_data(self) ->str:
        s = super().get_meta_data() + '\n'
        s += "X Normalization: Linear between [0..1]\n" \
             "Y Normalization: Linear between [0..1]\n" \
             "Classifier: Normalizing Spline Flow"
        return s
