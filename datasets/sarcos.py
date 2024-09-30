import os
from os.path import exists
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from core.data import GaussianNoise, VectorToTensor
from sklearn.datasets import load_svmlight_file
from core.data import BaseDataset, VectorDataset, normalize, postprocess_svm_data
import requests
import pandas as pd
import numpy as np
from scipy.io.arff import loadarff

class Sarcos(BaseDataset):

    def __init__(self, cache_folder:str, config:dict, pool_rng, encoded:bool,
                 data_file="sarcos_al.pt"):
        self.arff_file = os.path.join(cache_folder, "sarcos_raw.arff")
        super().__init__(cache_folder, config, pool_rng, encoded, data_file)


    def _download_data(self, target_to_one_hot=True):
        train_url = "http://old.openml.org/data/download/22102750/data.arff"

        if not exists(self.arff_file):
            r = requests.get(train_url)
            with open(self.arff_file, 'wb') as f:
                f.write(r.content)
            print("Download successful")

        if exists(self.arff_file):
            import arff
            rows = arff.load(self.arff_file)
            df = pd.DataFrame.from_records(rows)
            del rows
            # df = pd.get_dummies(df, columns=[1, 2, 3], prefix="id", dtype=float)
            y = df[21].values.astype(float)
            x = df.drop([21, 22, 23, 24, 25, 26, 27], axis=1).values
            del df

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
             "Y Normalization: Linear between [0..1]"
        return s
